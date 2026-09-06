"""A minimal, dependency-free glTF-binary (.glb) reader.

Only what the holographic viewer needs is parsed: the node hierarchy, triangle
topology and vertex positions, plus each primitive's base colour so a model can
be tinted the way its author shaded it. Textures, animation, skins and morph
targets are skipped outright -- the viewer draws wireframe, so a 4 MB planet
collapses to a few thousand line segments and the texture payload is dead
weight.

Two decisions keep loading fast enough to do at startup:

  * The binary chunk is read by seeking to the byte ranges the accessors
    actually name, so a file whose bulk is PNG texture data is never pulled
    into memory.
  * Every load is cached as a compressed .npz keyed by the source file's size
    and mtime. The parse only happens once per model, ever.

The ORBIX exports are the target: glTF 2.0, uncompressed, no
KHR_draco_mesh_compression. A file requiring an extension we cannot honour
raises GlbError rather than returning silently wrong geometry.
"""
import json
import os
import struct

import numpy as np

_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

# glTF componentType -> numpy dtype. All little-endian by spec.
_DTYPE = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
          "MAT2": 4, "MAT3": 9, "MAT4": 16}

# Extensions that only affect shading, so ignoring them still yields correct
# geometry. Anything else in extensionsRequired is a hard failure.
_HARMLESS = {
    "KHR_materials_emissive_strength", "KHR_materials_unlit",
    "KHR_materials_specular", "KHR_materials_ior", "KHR_materials_clearcoat",
    "KHR_materials_sheen", "KHR_materials_transmission",
    "KHR_texture_transform", "KHR_materials_volume",
    "KHR_materials_variants", "KHR_lights_punctual",
}

_TRIANGLES = 4


class GlbError(RuntimeError):
    pass


class Mesh:
    """Triangle soup in a unit-radius, origin-centred frame.

    `verts` is (V,3) float32 with |v| <= 1, `faces` is (F,3) int32, and
    `face_color` is (F,3) float32 BGR in 0..1 taken from each primitive's
    material. Storing colour per face rather than per primitive lets the
    renderer sort every triangle in the model together by depth.
    """

    __slots__ = ("verts", "faces", "face_color", "name", "extent")

    def __init__(self, verts, faces, face_color, name="", extent=1.0):
        self.verts = verts
        self.faces = faces
        self.face_color = face_color
        self.name = name
        self.extent = extent

    @property
    def n_tris(self):
        return int(len(self.faces))


# --------------------------------------------------------------- accessors --
def _read_accessor(fh, gltf, bin_offset, index):
    acc = gltf["accessors"][index]
    n = int(acc["count"])
    ncomp = _NCOMP[acc["type"]]
    dtype = _DTYPE.get(acc["componentType"])
    if dtype is None:
        raise GlbError("unsupported componentType %r" % (acc["componentType"],))
    itemsize = np.dtype(dtype).itemsize
    packed = ncomp * itemsize

    if "bufferView" not in acc:
        # Spec-legal: an accessor with no view reads as all zeros.
        return np.zeros((n, ncomp), np.float32)

    bv = gltf["bufferViews"][acc["bufferView"]]
    if bv.get("buffer", 0) != 0:
        raise GlbError("external buffers are not supported")
    start = (bin_offset + int(bv.get("byteOffset", 0))
             + int(acc.get("byteOffset", 0)))
    stride = int(bv.get("byteStride", 0)) or packed

    if stride == packed:
        fh.seek(start)
        raw = fh.read(packed * n)
        if len(raw) < packed * n:
            raise GlbError("accessor runs past the end of the binary chunk")
        out = np.frombuffer(raw, dtype=dtype, count=n * ncomp)
        out = out.reshape(n, ncomp)
    else:
        # Interleaved vertex buffer: pull the whole span once, then gather the
        # component bytes out of it rather than issuing n seeks.
        fh.seek(start)
        raw = fh.read(stride * (n - 1) + packed)
        if len(raw) < stride * (n - 1) + packed:
            raise GlbError("accessor runs past the end of the binary chunk")
        buf = np.frombuffer(raw, dtype=np.uint8)
        idx = (np.arange(n)[:, None] * stride
               + np.arange(packed)[None, :]).ravel()
        gathered = buf[idx].tobytes()
        out = np.frombuffer(gathered, dtype=dtype, count=n * ncomp)
        out = out.reshape(n, ncomp)

    if acc.get("normalized") and dtype != np.float32:
        info = np.iinfo(dtype)
        return out.astype(np.float32) / float(max(abs(info.min), info.max))
    return out


# ------------------------------------------------------------------ nodes --
def _trs_matrix(node):
    if "matrix" in node:
        # glTF stores matrices column-major.
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "rotation" in node:
        x, y, z, w = (float(v) for v in node["rotation"])
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag([float(v) for v in node["scale"]])
    if "translation" in node:
        m[:3, 3] = [float(v) for v in node["translation"]]
    return m


def _material_color(gltf, prim):
    """Base colour of a primitive's material, as BGR 0..1."""
    mi = prim.get("material")
    mats = gltf.get("materials") or []
    if mi is None or mi >= len(mats):
        return (1.0, 1.0, 1.0)
    mat = mats[mi]
    pbr = mat.get("pbrMetallicRoughness") or {}
    rgba = pbr.get("baseColorFactor")
    if rgba is None:
        # An unlit or purely emissive material still carries a usable hue.
        rgba = list(mat.get("emissiveFactor") or ())
    if rgba is None or len(rgba) < 3:
        return (1.0, 1.0, 1.0)
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    peak = max(r, g, b)
    if peak < 0.04:
        # Near-black reads as an invisible wireframe; treat it as untinted.
        return (1.0, 1.0, 1.0)
    if peak > 1.0:                      # emissive strength can exceed 1
        r, g, b = r / peak, g / peak, b / peak
    return (b, g, r)


# ------------------------------------------------------------------- load --
def load_glb(path, max_tris=6000, seed=11):
    """Parse `path` into a normalised Mesh, decimating past `max_tris`."""
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12:
            raise GlbError("file is too short to be a .glb")
        magic, version, _length = struct.unpack("<III", head)
        if magic != _MAGIC:
            raise GlbError("not a binary glTF (bad magic)")
        if version != 2:
            raise GlbError("glTF version %d is not supported" % version)

        gltf = None
        bin_offset = None
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            clen, ctype = struct.unpack("<II", hdr)
            here = fh.tell()
            if ctype == _CHUNK_JSON:
                gltf = json.loads(fh.read(clen).decode("utf-8"))
            elif ctype == _CHUNK_BIN:
                bin_offset = here
            fh.seek(here + clen + (-clen % 4))

        if gltf is None:
            raise GlbError("no JSON chunk")
        unmet = set(gltf.get("extensionsRequired") or ()) - _HARMLESS
        if unmet:
            raise GlbError("requires unsupported extension(s): "
                           + ", ".join(sorted(unmet)))
        if bin_offset is None:
            raise GlbError("no binary chunk (external .bin is not supported)")

        verts, faces, colors = _gather(fh, gltf, bin_offset)

    if not faces:
        raise GlbError("no triangle geometry found")

    V = np.concatenate(verts, axis=0).astype(np.float32)
    F = np.concatenate(faces, axis=0).astype(np.int64)
    C = np.concatenate(colors, axis=0).astype(np.float32)
    # A malformed index buffer must not become an out-of-bounds read later.
    F = F[(F >= 0).all(axis=1) & (F < len(V)).all(axis=1)]
    C = C[:len(F)]
    if len(F) == 0:
        raise GlbError("triangle indices are out of range")

    if len(F) > max_tris:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(F), max_tris, replace=False))
        F, C = F[keep], C[keep]
        # Drop vertices no surviving face references, so projection stays cheap.
        used = np.unique(F)
        remap = np.full(len(V), -1, np.int64)
        remap[used] = np.arange(len(used))
        V, F = V[used], remap[F]

    # Normalise into a unit ball around the centre of the bounding box.
    lo, hi = V.min(axis=0), V.max(axis=0)
    V = V - (lo + hi) * 0.5
    radius = float(np.linalg.norm(V, axis=1).max())
    extent = float(np.max(hi - lo))
    if radius > 1e-9:
        V = V / radius

    name = os.path.splitext(os.path.basename(path))[0]
    return Mesh(V.astype(np.float32), F.astype(np.int32), C, name, extent)


def _gather(fh, gltf, bin_offset):
    """Walk the scene graph, returning per-primitive verts / faces / colours."""
    nodes = gltf.get("nodes") or []
    meshes = gltf.get("meshes") or []
    scenes = gltf.get("scenes") or []
    scene = scenes[gltf.get("scene", 0)] if scenes else None
    roots = list(scene.get("nodes", ())) if scene else list(range(len(nodes)))

    verts, faces, colors = [], [], []
    base = 0
    stack = [(i, np.eye(4)) for i in reversed(roots)]
    seen = set()
    while stack:
        ni, parent = stack.pop()
        if ni in seen or ni >= len(nodes):     # guards against a cyclic file
            continue
        seen.add(ni)
        node = nodes[ni]
        world = parent @ _trs_matrix(node)
        for child in node.get("children", ()):
            stack.append((child, world))

        mi = node.get("mesh")
        if mi is None or mi >= len(meshes):
            continue
        for prim in meshes[mi].get("primitives", ()):
            if prim.get("mode", _TRIANGLES) != _TRIANGLES:
                continue
            pos_i = (prim.get("attributes") or {}).get("POSITION")
            if pos_i is None:
                continue
            p = _read_accessor(fh, gltf, bin_offset, pos_i)
            if p.shape[1] < 3:
                continue
            p = np.asarray(p[:, :3], np.float64) @ world[:3, :3].T + world[:3, 3]

            if "indices" in prim:
                idx = _read_accessor(fh, gltf, bin_offset,
                                     prim["indices"]).ravel()
            else:
                idx = np.arange(len(p))
            n = (len(idx) // 3) * 3
            if n == 0:
                continue
            tri = np.asarray(idx[:n], np.int64).reshape(-1, 3) + base

            verts.append(p)
            faces.append(tri)
            colors.append(np.tile(np.asarray(_material_color(gltf, prim),
                                             np.float32), (len(tri), 1)))
            base += len(p)
    return verts, faces, colors


# ------------------------------------------------------------------ cache --
def load_cached(path, cache_dir, max_tris=6000):
    """load_glb with an .npz cache invalidated by the source's size + mtime."""
    st = os.stat(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    key = "%s-%d-%d-%d" % (stem, st.st_size, int(st.st_mtime), max_tris)
    cache = os.path.join(cache_dir, key.replace(" ", "_") + ".npz")
    if os.path.exists(cache):
        try:
            with np.load(cache) as z:
                return Mesh(z["verts"], z["faces"], z["face_color"],
                            str(z["name"]), float(z["extent"]))
        except Exception:
            pass    # a truncated cache file just means we parse again

    mesh = load_glb(path, max_tris)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, verts=mesh.verts, faces=mesh.faces,
                            face_color=mesh.face_color, name=mesh.name,
                            extent=mesh.extent)
    except OSError:
        pass        # read-only install: loading still works, just slower
    return mesh

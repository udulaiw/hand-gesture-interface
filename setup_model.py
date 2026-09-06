"""One-time setup: the MediaPipe hand landmark model, and the ORBIX caches.

mediapipe >= 1.0 ships no bundled model, so the .task bundle has to be fetched
once. After this the application runs entirely offline.

The same run also pre-bakes every .glb in assets/orbix into the decimated .npz
the holographic viewer actually loads. That parse would otherwise happen the
first time each model is shown, which is a visible stutter at exactly the
wrong moment; doing it here makes every model instant from the first launch.

    python setup_model.py
"""
import os
import sys
import time
import urllib.request

import config


def download(url=config.MODEL_URL, dest=config.MODEL_PATH):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"[ok] model already present: {dest} "
              f"({os.path.getsize(dest) / 1e6:.1f} MB)")
        return 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[..] fetching {url}")

    def hook(blocks, block_size, total):
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = 100.0 * done / total
        sys.stdout.write(f"\r[..] {done / 1e6:5.1f} / {total / 1e6:.1f} MB  "
                         f"{pct:5.1f}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, hook)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"\n[!!] download failed: {e}")
        print("     You can also download the file manually and place it at:")
        print(f"     {dest}")
        return 1

    print()
    size = os.path.getsize(tmp)
    if size < 1_000_000:
        os.remove(tmp)
        print(f"[!!] downloaded file is only {size} bytes — not a model bundle.")
        return 1
    os.replace(tmp, dest)
    print(f"[ok] saved {dest} ({size / 1e6:.1f} MB)")
    return 0


def prebake():
    """Parse every configured model once so the viewer only reads caches."""
    from utils.glb import GlbError, load_cached

    ok = failed = 0
    for stem, label, _accent in config.ORBIX_MODELS:
        path = os.path.join(config.ORBIX_DIR, stem + ".glb")
        if not os.path.exists(path):
            print(f"[--] {label:<12} missing: {path}")
            failed += 1
            continue
        t0 = time.perf_counter()
        try:
            mesh = load_cached(path, config.ORBIX_CACHE, config.ORBIX_MAX_TRIS)
        except (GlbError, OSError, ValueError, KeyError) as e:
            print(f"[!!] {label:<12} {e}")
            failed += 1
            continue
        ok += 1
        print(f"[ok] {label:<12} {mesh.n_tris:5d} tris  {len(mesh.verts):5d} "
              f"verts  {(time.perf_counter() - t0) * 1000:6.1f} ms")
    print(f"[ok] {ok} model(s) cached in {config.ORBIX_CACHE}"
          + (f", {failed} unavailable" if failed else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    code = download()
    print()
    print("[..] pre-baking ORBIX model caches")
    code = prebake() or code
    sys.exit(code)

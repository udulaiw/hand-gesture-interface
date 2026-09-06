@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt > install_log.txt 2>&1
rem One-time: fetch the landmark model and pre-bake the ORBIX model caches.
python setup_model.py >> install_log.txt 2>&1
python main.py > run_log.txt 2>&1
echo DONE > run_done.txt

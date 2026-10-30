@echo off
cd /d "%~dp0"
python hunter.py >> data\last_run_output.txt 2>&1

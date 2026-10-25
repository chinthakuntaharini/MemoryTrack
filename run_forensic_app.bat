@echo off
REM Launcher for Forensic Non-Facial Person Tracker & Re-ID Desktop Application
echo Starting Forensic Re-ID Surveillance Application...
python -m forensic_app.main
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)

@echo off
echo ===================================================
echo MemoryTrack - Build Windows Executable
echo ===================================================
echo.
echo Building standalone package with MemoryTrack.spec (UPX disabled for PyTorch compatibility)...
echo This will package PyTorch, OpenCV, MediaPipe, and YOLO.
echo.
python -m PyInstaller --clean --noconfirm MemoryTrack.spec

echo.
echo ===================================================
echo Build complete! 
echo You can find the executable in the 'dist\MemoryTrack' folder.
echo Double click 'dist\MemoryTrack\MemoryTrack.exe' to run.
echo ===================================================
pause

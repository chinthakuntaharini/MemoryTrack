# MemoryTrack.spec
# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Collect all dynamic assets, binaries, and hidden submodules
torch_datas, torch_binaries, torch_hidden = collect_all('torch')
torchvision_datas, torchvision_binaries, torchvision_hidden = collect_all('torchvision')
ultralytics_datas, ultralytics_binaries, ultralytics_hidden = collect_all('ultralytics')
mediapipe_datas, mediapipe_binaries, mediapipe_hidden = collect_all('mediapipe')

a = Analysis(
    ['desktop_app.py'],
    pathex=['.'],
    binaries=torch_binaries + torchvision_binaries + ultralytics_binaries + mediapipe_binaries,
    datas=[
        ('config', 'config'),
        ('database', 'database'),
        ('core', 'core'),
        ('utils', 'utils'),
        ('yolo11n.pt', '.'),
        ('yolo11n-pose.pt', '.'),
    ] + torch_datas + torchvision_datas + ultralytics_datas + mediapipe_datas,
    hiddenimports=[
        # PyQt6
        'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui',
        # Torch & Vision
        'torch', 'torch.nn', 'torch.nn.functional', 'torchvision',
        # Ultralytics / YOLO
        'ultralytics', 'ultralytics.models', 'ultralytics.engine',
        'ultralytics.utils', 'ultralytics.data', 'lapx',
        # MediaPipe
        'mediapipe', 'mediapipe.python',
        # OpenCV
        'cv2',
        # FAISS
        'faiss',
        # NumPy, SciPy
        'numpy', 'scipy', 'sklearn',
        # DB
        'sqlalchemy',
        # Misc
        'yaml', 'pyyaml', 'PIL',
    ] + torch_hidden + torchvision_hidden + ultralytics_hidden + mediapipe_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude web frameworks to save space
        'streamlit', 'altair', 'plotly', 'tensorboard',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MemoryTrack',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX - UPX corrupts PyTorch DLLs (c10.dll, torch_cpu.dll)
    console=False,  # Native GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # UPX must be false to prevent WinError 1114
    upx_exclude=['*'],
    name='MemoryTrack',
)

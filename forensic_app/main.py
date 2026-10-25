"""
Forensic Person Tracker & Re-ID Desktop Application Entrypoint
Initializes PyQt6 runtime with crash guards and global exception traps.
"""

import os
import sys

# Critical configuration for PyTorch / OpenMP on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

# Handle Windows DLL directory registration
if sys.platform == "win32":
    try:
        import site
        for site_pkg in site.getsitepackages():
            torch_lib = os.path.join(site_pkg, "torch", "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        # Also check user site packages
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            torch_lib = os.path.join(user_site, "torch", "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# Critical: MUST import torch BEFORE PyQt6 on Windows to prevent WinError 1114 DLL collision
try:
    import torch
except Exception as e:
    print(f"Warning loading PyTorch: {e}")

import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ForensicApp")

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from forensic_app.ui.main_window import ForensicMainWindow


def global_exception_hook(exctype, value, tb):
    """Crash trap: prevent application from silently closing on uncaught errors."""
    import traceback
    err_text = "".join(traceback.format_exception(exctype, value, tb))
    logger.critical(f"Uncaught Exception: {err_text}")
    print(err_text, file=sys.stderr)

    # Show error modal if QApplication instance exists
    if QApplication.instance():
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Application Diagnostic Trap")
        msg_box.setText(f"A recoverable exception occurred:\n\n{value}")
        msg_box.setDetailedText(err_text)
        msg_box.exec()


def main():
    """Launch forensic PyQt6 desktop application."""
    sys.excepthook = global_exception_hook

    app = QApplication(sys.argv)
    app.setApplicationName("Forensic Non-Facial Re-ID Tracker")
    app.setApplicationVersion("2.0.0")

    window = ForensicMainWindow()
    window.show()

    logger.info("Forensic Re-ID Application running.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

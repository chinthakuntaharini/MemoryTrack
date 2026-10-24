import os
import sys
from pathlib import Path

def create_desktop_shortcut():
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    shortcut_path = desktop / "MemoryTrack.lnk"
    
    project_dir = Path(__file__).parent.parent.resolve()
    target_exe = project_dir / "dist" / "MemoryTrack" / "MemoryTrack.exe"
    target_path = target_exe if target_exe.exists() else (project_dir / "MemoryTrack_Launcher.bat")
    work_dir = target_exe.parent if target_exe.exists() else project_dir
    
    vbs_content = f'''
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{target_path}"
oLink.WorkingDirectory = "{work_dir}"
oLink.Description = "MemoryTrack AI Missing Person Tracking System"
oLink.Save
'''
    vbs_file = project_dir / "create_shortcut.vbs"
    with open(vbs_file, "w") as f:
        f.write(vbs_content)
        
    os.system(f'cscript //nologo "{vbs_file}"')
    if vbs_file.exists():
        vbs_file.unlink()
        
    print(f"[OK] Desktop shortcut created: {shortcut_path}")

if __name__ == "__main__":
    create_desktop_shortcut()

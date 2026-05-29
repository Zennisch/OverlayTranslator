import os
import sys
from pathlib import Path

# When packaged/frozen under PyInstaller, sys.frozen is True and sys._MEIPASS contains the runtime temporary folder path.
# We must insert it to sys.path to resolve absolute imports cleanly.
if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass and meipass not in sys.path:
        sys.path.insert(0, meipass)

    app_dir = os.path.join(meipass, "app")
    if os.path.exists(app_dir) and app_dir not in sys.path:
        sys.path.insert(0, app_dir)
else:
    # Development mode: Add project root and app root to sys.path
    project_root = Path(__file__).resolve().parents[1]
    app_root = project_root / "app"

    project_root_str = str(project_root)
    app_root_str = str(app_root)

    if app_root_str not in sys.path:
        sys.path.insert(0, app_root_str)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

import os
import sys


def add_sys_path(path):
    if path:
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)


if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", "")
    app_dir = os.path.join(meipass, "app")

    add_sys_path(meipass)
    add_sys_path(app_dir)

else:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_root = os.path.join(project_root, "app")

    add_sys_path(project_root)
    add_sys_path(app_root)

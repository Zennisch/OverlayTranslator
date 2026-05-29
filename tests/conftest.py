import sys
from pathlib import Path

# Automatically add the project root directory and app directory to sys.path so pytest can locate 'app' cleanly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

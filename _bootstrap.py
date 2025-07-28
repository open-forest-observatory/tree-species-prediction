# _bootstrap.py

# *** DON'T RUN THIS DIRECTLY ***
# Run symlink_bootstrap.py to link this bootstap to working dirs (dirs with scripts ran directly)

# handle path issues so scripts run in subfolders can see config files and util scripts,
# as well as adding other python interpreters to path instead of needing to do it repeatedly.

# *** be sure to import _bootstrap BEFORE any relative imports ***

# e.g. inside of `1_data_prep/1_get_mission_altitude_driver_script.py`:
# import _bootstrap
# from configs.path_config import path_config

import os
import sys
from pathlib import Path

# ----- Resolve relative imports (e.g. `config/`) -----

here = Path(__file__).parent.resolve() # current dir
project_rt = here.parent # project root

# ensure root is on sys.path
root_str = str(project_rt)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# add every first‑level folder under root
for sub in project_rt.iterdir():
    if sub.is_dir() and not sub.name.startswith('.'):
        sub_str = str(sub.resolve())
        if sub_str not in sys.path:
            sys.path.insert(0, sub_str)


# ----- Add alt Python script dirs to path (e.g. Automate Metashape) -----
try:
    from configs.path_config import path_config
except ImportError:
    raise FileNotFoundError("ERROR: Could not find path config to add interpreters to system path")

# ADD OTHER SCRIPT DIRECTORIES HERE AS NEEDED
SCRIPTS_PATHS = [
    path_config.automate_metashape_path / "python",
]

for script_path in SCRIPTS_PATHS:
    if script_path.exists():
        sys.path.insert(0, str(script_path))


# RUN THIS ONCE TO SYMLINK _bootstrap.py into runnable script folders
import sys
from pathlib import Path

# list of sub directories of the project containing scripts that are run directly
# NOTE: do NOT need to add folders like `utils/` or `config/` since they are never directly called
WORKING_DIRS = [
    "1_data_prep",
    "2_training",
]

BOOTSTRAP_NAME = "_bootstrap.py"

def main():
    root = Path(__file__).parent.resolve()
    bootstrap = root / BOOTSTRAP_NAME

    if not bootstrap.exists():
        print(f"Missing {BOOTSTRAP_NAME} in project root ({root})")
        sys.exit(1)

    for subname in WORKING_DIRS:
        subdir = root / subname
        if not subdir.is_dir():
            print(f"Skipping '{subname}': not found or not a directory")
            continue

        link = subdir / BOOTSTRAP_NAME
        # remove any existing file/symlink
        if link.exists() or link.is_symlink():
            link.unlink()

        # create a relative symlink: subdir/_bootstrap.py -> ../_bootstrap.py
        try:
            link.symlink_to(bootstrap)
            print(f"Link Created: {link} -> {bootstrap}")
        except Exception as e:
            print(f"Failed to link into '{subname}':\n{e}")

if __name__ == "__main__":
    main()
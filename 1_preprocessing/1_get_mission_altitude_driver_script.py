import sys
import subprocess
import re
import tempfile
from pathlib import Path
from tqdm import tqdm

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (
    ALL_MISSIONS_REMOTE_FOLDER,
    MISSION_ALTITUDES_FOLDER,
    MISSIONS_OUTSIDE_DTM_LIST,
)

# List to track failed missions
failed_missions = []

# List all folders from remote
list_cmd = ["rclone", "lsf", ALL_MISSIONS_REMOTE_FOLDER]
result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
# Determine mission IDs to evaluate
mission_ids = [
    line.strip("/") for line in result.stdout.splitlines() if re.match(r"\d+", line)
]

# Iterate through folders
for mission_id in tqdm(mission_ids):
    mission_id_folder = f"{mission_id}_01"
    base_remote_path = f"{ALL_MISSIONS_REMOTE_FOLDER}/{mission_id}/processed_01/full"
    camera_file = f"{mission_id_folder}_cameras.xml"
    dtm_file = f"{mission_id_folder}_dtm-ptcloud.tif"
    output_csv = MISSION_ALTITUDES_FOLDER / f"{mission_id_folder}_altitude_summary.csv"

    # Skip already processed missions
    if output_csv.exists():
        continue

    # Remote file paths
    camera_remote = f"{base_remote_path}/{camera_file}"
    dtm_remote = f"{base_remote_path}/{dtm_file}"

    try:
        tmp_camera = tempfile.NamedTemporaryFile(suffix=".xml")
        tmp_dtm = tempfile.NamedTemporaryFile(suffix=".tif")

        camera_local = Path(tmp_camera.name)
        dtm_local = Path(tmp_dtm.name)

        # Download files to temporary paths
        subprocess.run(
            ["rclone", "copyto", camera_remote, str(camera_local)], check=True
        )
        subprocess.run(["rclone", "copyto", dtm_remote, str(dtm_local)], check=True)

        # Run script to get the mission altitude
        subprocess.run(
            [
                "python",
                "1_get_mission_altitude.py",
                "--camera-file",
                str(camera_local),
                "--dtm-file",
                str(dtm_local),
                "--output-csv",
                str(output_csv),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    except Exception as e:
        # Writes appropriate error to file. This is for missions that have missing files or
        # failed computing mission altitude because of the ValueError due to >10% of camera points outside DTM.
        failed_missions.append((mission_id, f"Error: {str(e)}"))

# Write failure log
if failed_missions:
    with MISSIONS_OUTSIDE_DTM_LIST.open("w") as f:
        for mid, reason in failed_missions:
            f.write(f"{mid},{reason}\n")
    print(f"Some missions failed. See '{MISSIONS_OUTSIDE_DTM_LIST}' for details.")
else:
    print("All missions processed successfully!")

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

import _bootstrap
from configs.path_config import path_config

path_config.mission_altitudes_folder.mkdir(parents=True, exist_ok=True)

# List to track failed missions
failed_missions = []

# List all folders from remote
list_cmd = ["rclone", "lsf", path_config.all_missions_remote_folder]
result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
# Determine mission IDs to evaluate
mission_ids = [
    line.strip("/") for line in result.stdout.splitlines() if re.match(r"\d+", line)
]

# Iterate through folders
for mission_id in tqdm(mission_ids):
    mission_id_folder = f"{mission_id}"
    base_remote_path = f"{path_config.all_missions_remote_folder}/{mission_id}/processed_02/full"
    camera_file = f"{mission_id_folder}_cameras.xml"
    dtm_file = f"{mission_id_folder}_dtm-ptcloud.tif"
    output_csv = path_config.mission_altitudes_folder / f"{mission_id_folder}_altitude_summary.csv"

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
    with path_config.missions_outside_dtm_list.open("w") as f:
        for mid, reason in failed_missions:
            f.write(f"{mid},{reason}\n")
    print(f"Some missions failed. See '{path_config.missions_outside_dtm_list}' for details.")
else:
    print("All missions processed successfully!")

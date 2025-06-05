import subprocess
import os
import shutil
import re
from tqdm import tqdm


remote = "js2s3:ofo-public/drone/missions_01"
local_tmp = "/ofo-share/scratch-amritha/tree-species-scratch/tmp"
output_dir = "/ofo-share/scratch-amritha/tree-species-scratch/mission_altitudes"
failed_log_path = "/ofo-share/scratch-amritha/tree-species-scratch/failed_missions.txt"


os.makedirs(local_tmp, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# List to track failed missions
failed_missions = []

# List all folders under remote
list_cmd = ["rclone", "lsf", remote]
result = subprocess.run(list_cmd, capture_output=True, text=True)

folders = [line.strip("/") for line in result.stdout.splitlines() if re.match(r"\d+", line)]
# Iterate through each folder and set paths to camera, dtm and output files
for folder in tqdm(folders):
    mission_id = f"{folder}_01"
    base_remote_path = f"{remote}/{folder}/processed_01/full"
    camera_file = f"{mission_id}_cameras.xml"
    dtm_file = f"{mission_id}_dtm-ptcloud.tif"
    output_csv = os.path.join(output_dir, f"{mission_id}_altitude_summary.csv")

    # Skip if output already exists
    if os.path.exists(output_csv):
        continue

    camera_remote = f"{base_remote_path}/{camera_file}"
    dtm_remote = f"{base_remote_path}/{dtm_file}"
    camera_local = os.path.join(local_tmp, camera_file)
    dtm_local = os.path.join(local_tmp, dtm_file)

    # Try downloading both files
    subprocess.run(["rclone", "copyto", camera_remote, camera_local], check=True)
    subprocess.run(["rclone", "copyto", dtm_remote, dtm_local], check=True)
    # Check if files exist after download
    if not (os.path.exists(camera_local) and os.path.exists(dtm_local)):
        print(f"Download failed for {mission_id}.")
        failed_missions.append((mission_id, "Missing files"))
        continue


    # Run the altitude calculation script
    try:
        subprocess.run([
            "python3", "1_get_mission_altitude.py",
            "--camera-file", camera_local,
            "--dtm-file", dtm_local,
            "--output-csv", output_csv
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        failed_missions.append((mission_id, "Altitude computation failed"))

    # Delete the temporary files
    for path in [camera_local, dtm_local]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

# Log failed missions
if failed_missions:
    with open(failed_log_path, "w") as f:
        for id, reason in failed_missions:
            f.write(f"{id},{reason}\n")
    print(f"Some missions failed. See '{failed_log_path}' for details.")
else:
    print("All missions processed successfully!")
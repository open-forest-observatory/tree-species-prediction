import subprocess
import os
import re
import tempfile
from tqdm import tqdm

# Path to parent remote folder with all missions
remote = "js2s3:ofo-public/drone/missions_01"
# Local paths to save outputs
output_dir = "/ofo-share/scratch-amritha/tree-species-scratch/mission_altitudes"
failed_log_path = "/ofo-share/scratch-amritha/tree-species-scratch/failed_missions.txt"

os.makedirs(output_dir, exist_ok=True)

# List to track failed missions
failed_missions = []

# List all folders from remote
list_cmd = ["rclone", "lsf", remote]
result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
folders = [line.strip("/") for line in result.stdout.splitlines() if re.match(r"\d+", line)]

# Iterate through folders
for folder in tqdm(folders):
    mission_id = f"{folder}_01"
    base_remote_path = f"{remote}/{folder}/processed_01/full"
    camera_file = f"{mission_id}_cameras.xml"
    dtm_file = f"{mission_id}_dtm-ptcloud.tif"
    output_csv = os.path.join(output_dir, f"{mission_id}_altitude_summary.csv")

    # Skip already processed missions
    if os.path.exists(output_csv):
        continue

    # Remote file paths
    camera_remote = f"{base_remote_path}/{camera_file}"
    dtm_remote = f"{base_remote_path}/{dtm_file}"

    try:
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_camera, \
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_dtm:

            camera_local = tmp_camera.name
            dtm_local = tmp_dtm.name

        # Download files to temporary paths
        subprocess.run(["rclone", "copyto", camera_remote, camera_local], check=True)
        subprocess.run(["rclone", "copyto", dtm_remote, dtm_local], check=True)

        # Confirm files exist
        if not (os.path.exists(camera_local) and os.path.exists(dtm_local)):
            failed_missions.append((mission_id, "Missing files"))
            continue
        
        # Run script to get the mission altitude
        subprocess.run([
            "python", "1_get_mission_altitude.py",
            "--camera-file", camera_local,
            "--dtm-file", dtm_local,
            "--output-csv", output_csv
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError:
        # During ValueError when more than 10% of camera points fall outside the DTM extent
        failed_missions.append((mission_id, "Altitude computation failed"))

    finally:
        # Clean up temporary files
        for tmp_path in [camera_local, dtm_local]:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

# Write failure log
if failed_missions:
    with open(failed_log_path, "w") as f:
        for mid, reason in failed_missions:
            f.write(f"{mid},{reason}\n")
    print(f"Some missions failed. See '{failed_log_path}' for details.")
else:
    print("All missions processed successfully!")
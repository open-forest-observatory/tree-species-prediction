import subprocess
import os
import re
import tempfile
from tqdm import tqdm

# Path to parent remote folder with all missions
remote = "js2s3:ofo-public/drone/missions_01"
# Local paths to save outputs
output_dir = "/ofo-share/scratch-amritha/tree-species-scratch/mission_altitudes2"
failed_log_path = "/ofo-share/scratch-amritha/tree-species-scratch/failed_missions2.txt"

os.makedirs(output_dir, exist_ok=True)

# List to track failed missions
failed_missions = []

# List all folders from remote
list_cmd = ["rclone", "lsf", remote]
result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
# Determine mission IDs to evaluate
mission_ids = [line.strip("/") for line in result.stdout.splitlines() if re.match(r"\d+", line)]

# Iterate through folders
for mission_id in tqdm(mission_ids):
    mission_id_folder = f"{mission_id}_01"
    base_remote_path = f"{remote}/{mission_id}/processed_01/full"
    camera_file = f"{mission_id_folder}_cameras.xml"
    dtm_file = f"{mission_id_folder}_dtm-ptcloud.tif"
    output_csv = os.path.join(output_dir, f"{mission_id_folder}_altitude_summary.csv")

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
        
        # Run script to get the mission altitude
        subprocess.run([
            "python", "1_get_mission_altitude.py",
            "--camera-file", camera_local,
            "--dtm-file", dtm_local,
            "--output-csv", output_csv
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except Exception as e:
        # Writes appropriate error to file. This is for missions that have missing files or 
        # failed computing mission altitude because of the ValueError due to >10% of camera points outside DTM.
        failed_missions.append((mission_id, f"Error: {str(e)}"))

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
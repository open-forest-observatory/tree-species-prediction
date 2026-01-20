import subprocess
import tempfile
from pathlib import Path

from _get_mission_altitude import compute_height_above_ground, get_camera_locations

import _bootstrap
from configs.path_config import path_config

METASHAPE_CONFIG = Path(
    path_config.automate_metashape_path, "config", "config-base.yml"
)


# This needs to download the single-mission photogrammetry camera results file and dtm.
# Then run altitude extraction using only the images from the subset identified in step 05
# Using these altitudes, we can determine the real height above ground for each mission and compute
# the appropriate offset between the two.
def download_camera_and_dtm(mission_id):
    base_remote_path = (
        f"{path_config.all_missions_remote_folder}/{mission_id}/processed_02/full"
    )
    camera_file = f"{mission_id}_cameras.xml"
    dtm_file = f"{mission_id}_dtm-ptcloud.tif"

    # Remote file paths
    camera_remote = f"{base_remote_path}/{camera_file}"
    dtm_remote = f"{base_remote_path}/{dtm_file}"

    tmp_camera = tempfile.NamedTemporaryFile(suffix=".xml")
    tmp_dtm = tempfile.NamedTemporaryFile(suffix=".tif")

    camera_local = Path(tmp_camera.name)
    dtm_local = Path(tmp_dtm.name)

    # Download files to temporary paths
    subprocess.run(["rclone", "copyto", camera_remote, str(camera_local)], check=True)
    subprocess.run(["rclone", "copyto", dtm_remote, str(dtm_local)], check=True)

    return (tmp_camera, tmp_dtm)


def compute_altitude(mission_id):
    tmp_camera, tmp_dtm = download_camera_and_dtm(mission_id)

    camera_local = Path(tmp_camera.name)
    dtm_local = Path(tmp_dtm.name)

    cam_locations = get_camera_locations(camera_local)
    camera_elevations = compute_height_above_ground(cam_locations, dtm_local)
    breakpoint()


if __name__ == "__main__":
    compute_altitude("000335")

from pathlib import Path
from typing import Optional, Union

import numpy as np
import rioxarray
from rasterio.enums import Resampling

import _bootstrap
from configs.path_config import path_config


def compute_CHM(
    dsm_path: Union[str, Path],
    dtm_path: Union[str, Path],
    output_chm_path: Union[str, Path],
    resolution: Optional[float] = None,
    clip_negative: bool = True,
):
    """Create a CHM by subtracting the DTM values from the DSM

    Args:
        dsm_path (Union[str, Path]):
            Path to read the DSM from
        dtm_path (Union[str, Path]):
            Path to read the DTM from
        output_chm_path (Union[str, Path]):
            Path to write the newly-computed CHM to
        resolution (Optional[float], optional):
            Spatial resolution of the CHM in meters. If unset, it will default to the resolution of
            the DSM. Defaults to None.
        clip_negative (Optional[bool], optional):
            Set all negative CHM values to 0. Defaults to True.
    """
    # Read the data
    dtm = rioxarray.open_rasterio(dtm_path, masked=True)
    dsm = rioxarray.open_rasterio(dsm_path, masked=True)
    if resolution is None:
        # No resampling should be applied
        dsm_resampled = dsm
    else:
        if dsm.rio.crs.is_geographic:
            raise ValueError(
                "Reprojection of a geographic CRS to meter units will fail"
            )
        # Determine the current resolution
        average_dsm_resolution = np.mean(np.abs(dsm.rio.resolution()))

        # Determine the new pixel width and heights
        scale_factor = average_dsm_resolution / resolution
        new_height = int(dsm.rio.height * scale_factor)
        new_width = int(dsm.rio.width * scale_factor)

        # Use an averaging filter for downsampling and bicubic for upsampling
        resampling = Resampling.average if scale_factor < 1 else Resampling.cubic
        # Perform the resampling
        dsm_resampled = dsm.rio.reproject(
            dsm.rio.crs, shape=(new_height, new_width), resampling=resampling
        )

    # Determine if the DTM has a higher resolution than the resampled DSM
    # TODO consider how to make this more robust to data that's in geographic coordinates
    dtm_higher_resolution = np.mean(np.abs(dtm.rio.resolution())) < np.mean(
        np.abs(dsm_resampled.rio.resolution())
    )
    # If the DSM is higher resolution it will be coarstened, so use averaging. Else use bicubic.
    resampling = Resampling.average if dtm_higher_resolution else Resampling.cubic
    # Pixel-align the DTM to the DSM
    dtm_resampled = dtm.rio.reproject_match(dsm_resampled, resampling=resampling)

    # Subtract the two products
    chm = dsm_resampled - dtm_resampled

    if clip_negative:
        # Set all negative values to zero
        chm = chm.clip(min=0)

    # Save to disk
    Path(output_chm_path).parent.mkdir(parents=True, exist_ok=True)
    chm.rio.to_raster(output_chm_path)


if __name__ == "__main__":
    if not path_config.photogrammetry_folder.is_symlink():
        # symlink from where argo produced the photogrammetry outputs to the working file tree
        path_config.photogrammetry_folder.symlink_to(
            path_config.photogrammetry_folder_argo
        )
    # List all the folders, corresponding to photogrammetry for a nadir-oblique pair
    photogrammetry_run_folders = path_config.photogrammetry_folder.glob("*_*")
    # Iterate over the folders
    for photogrammetry_run_folder in photogrammetry_run_folders:
        # The last part of the path is the <nadir id>_<oblique id> pair
        run_ID = photogrammetry_run_folder.parts[-1]

        # This is where all the data products are saved to
        photogrammetry_products_folder = Path(photogrammetry_run_folder, "output")
        # There should be only one file with the corresponding ending, but the first part is a
        # timestamp that is unknown
        dsm_file = Path(photogrammetry_products_folder, f"{run_ID}_dsm-mesh.tif")
        dtm_file = Path(photogrammetry_products_folder, f"{run_ID}_dtm-ptcloud.tif")

        if not dsm_file.is_file() or not dtm_file.is_file():
            print(f"Skipping run {run_ID} because of missing data")
            continue

        # All the outputs will be saved to one folder
        output_chm_path = Path(path_config.chm_folder, f"{run_ID}.tif")
        # Run the computation
        compute_CHM(
            dsm_path=dsm_file, dtm_path=dtm_file, output_chm_path=output_chm_path
        )

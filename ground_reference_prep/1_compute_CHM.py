from pathlib import Path
from typing import Optional, Union
import sys

from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import rioxarray
from rasterio.enums import Resampling
import rasterio as rio

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import PHOTOGRAMMETRY_FOLDER, CHM_FOLDER,GROUND_REFERENCE_PLOTS_FILE

def reproject_to_match(source_raster, target_raster, resampling):
    destination = np.full(target_raster.shape, fill_value=np.nan)
    destination, dst_transform = rio.warp.reproject(
        source_raster.read(1),
        destination=destination,
        src_transform=source_raster.transform,
        src_crs=source_raster.crs,
        dst_transform=target_raster.transform,
        dst_crs=target_raster.crs,
        dst_nodata=target_raster.nodata,
        dst_resolution=target_raster.res,
        resampling=resampling,
    )
    return destination

def compute_CHM(
    dsm_path: Union[str, Path],
    dtm_path: Union[str, Path],
    output_chm_path: Union[str, Path],
    resolution: Optional[float] = None,
    clip_negative: bool = True,
    spatial_crop_bounds = None,
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
    dtm = rio.open(dtm_path, masked=True)
    dsm = rio.open(dsm_path, masked=True)
    if resolution is None:
        # No resampling should be applied
        dsm_resampled = dsm
    else:
        if dsm.rio.crs.is_geographic:
            raise ValueError(
                "Reprojection of a geographic CRS to meter units will fail"
            )
        # Determine the current resolution
        average_dsm_resolution = np.mean(np.abs(dsm.res))

        # Determine the new pixel width and heights
        scale_factor = average_dsm_resolution / resolution
        new_height = int(dsm.rio.height * scale_factor)
        new_width = int(dsm.rio.width * scale_factor)

        # Use an averaging filter for downsampling and bicubic for upsampling
        resampling = Resampling.average if scale_factor < 1 else Resampling.cubic
        # Perform the resampling
        dsm_resampled = dsm.reproject(
            dsm.crs, shape=(new_height, new_width), resampling=resampling
        )

    # Determine if the DTM has a higher eesolution than the resampled DSM
    # TODO consider how to make this more robust to data that's in geographic coordinates
    dtm_higher_resolution = np.mean(np.abs(dtm.res)) < np.mean(
        np.abs(dsm_resampled.res)
    )
    # If the DSM is higher resolution it will be coarstened, so use averaging. Else use bicubic.
    resampling = Resampling.average if dtm_higher_resolution else Resampling.cubic
    # Pixel-align the DTM to the DSM
    dtm_resampled = reproject_to_match(dtm, dsm_resampled, resampling=resampling)
    #dtm.reproject_match(dsm_resampled, resampling=resampling)

    dsm_data = dsm_resampled.read(1)
    breakpoint()

    # Subtract the two products
    chm = dsm_data - dtm_resampled

    if clip_negative:
        # Set all negative values to zero
        chm = chm.clip(min=0)

    if spatial_crop_bounds is not None:
        chm = mask(chm, spatial_crop_bounds.geometry, crop=True)

    # Save to disk
    Path(output_chm_path).parent.mkdir(parents=True, exist_ok=True)
    chm.to_raster(output_chm_path)


if __name__ == "__main__":
    # List all the folders, corresponding to photogrammetry for a nadir-oblique pair
    photogrammetry_run_folders = PHOTOGRAMMETRY_FOLDER.glob("*_*")

    # Load the plot bounds data
    all_plot_bounds = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)

    # Iterate over the folders
    for photogrammetry_run_folder in photogrammetry_run_folders:
        # The last part of the path is the <nadir id>_<oblique id> pair
        run_ID = photogrammetry_run_folder.parts[-1]

        plot_ID = run_ID.split("_")[0]
        plot_bounds = all_plot_bounds.query("plot_id == @plot_ID")

        # This is where all the data products are saved to
        photogrammetry_products_folder = Path(photogrammetry_run_folder, "outputs")
        # There should be only one file with the corresponding ending, but the first part is a
        # timestamp that is unknown
        dsm_file = Path(photogrammetry_run_folder, "outputs", f"{run_ID}_dsm-mesh.tif")
        dtm_file = Path(photogrammetry_run_folder, "outputs", f"{run_ID}_dtm-ptcloud.tif")

        if not dsm_file.is_file() or not dtm_file.is_file():
            print(f"Skipping run {run_ID} because of missing data")
            continue

        # All the outputs will be saved to one folder
        output_chm_path = Path(CHM_FOLDER, f"{run_ID}.tif")
        # Run the computation
        compute_CHM(
            dsm_path=dsm_file, dtm_path=dtm_file, output_chm_path=output_chm_path, spatial_crop_bounds=plot_bounds
        )

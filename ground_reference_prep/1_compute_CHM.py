from pathlib import Path

import numpy as np
import rioxarray
from rasterio.enums import Resampling


def compute_CHM(dsm_path, dtm_path, output_chm_path, resolution=None):
    dtm = rioxarray.open_rasterio(dtm_path, masked=True)
    dsm = rioxarray.open_rasterio(dsm_path, masked=True)

    if resolution is not None:
        # Not resampling should be applied
        dtm_resampled = dtm
    else:
        # TODO consider a check to ensure the data is in a projected CRS
        # Determine the current resolution
        dtm_resolution = dtm.rio.resolution()
        average_res = np.mean(np.abs(dtm_resolution))

        # Determine the new pixel width and heights
        upscale_factor = average_res / resolution
        new_height = int(dtm.rio.height * upscale_factor)
        new_width = int(dtm.rio.width * upscale_factor)

        # Perform the resampling
        dtm_resampled = dtm.rio.reproject(
            dtm.rio.crs, shape=(new_height, new_width), resampling=Resampling.bilinear
        )

    # Pixel-align the dsm to the DTM
    dsm_resampled = dsm.rio.reproject_match(dtm_resampled)

    # Subtract the two products
    chm = dsm_resampled - dtm_resampled
    # Set all negative values to zero
    chm = chm.clip(min=0)

    # Save to disk
    Path(output_chm_path).parent.mkdir(parents=True, exist_ok=True)
    chm.rio.to_raster(output_chm_path)

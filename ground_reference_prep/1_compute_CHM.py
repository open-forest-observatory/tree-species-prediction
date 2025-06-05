from pathlib import Path
from typing import Optional, Union

import numpy as np
import rioxarray
from rasterio.enums import Resampling


def compute_CHM(
    dsm_path: Union[str, Path],
    dtm_path: Union[str, Path],
    output_chm_path: Union[str, Path],
    resolution: Optional[float] = None,
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
            the DTM. Defaults to None.
    """
    # Read the data
    dtm = rioxarray.open_rasterio(dtm_path, masked=True)
    dsm = rioxarray.open_rasterio(dsm_path, masked=True)

    if resolution is None:
        # No resampling should be applied
        dtm_resampled = dtm
    else:
        if dtm.rio.crs.is_geographic:
            raise ValueError(
                "Reprojection of a geographic CRS to meter units will fail"
            )
        # Determine the current resolution
        dtm_resolution = dtm.rio.resolution()
        average_res = np.mean(np.abs(dtm_resolution))

        # Determine the new pixel width and heights
        scale_factor = average_res / resolution
        new_height = int(dtm.rio.height * scale_factor)
        new_width = int(dtm.rio.width * scale_factor)

        # Use an averaging filter for downsampling and bicubic for upsampling
        resampling = Resampling.average if scale_factor < 1 else Resampling.cubic
        # Perform the resampling
        dtm_resampled = dtm.rio.reproject(
            dtm.rio.crs, shape=(new_height, new_width), resampling=resampling
        )

    # Determine if the DSM has a higher resolution than the resampled DTM
    # TODO consider how to make this more robust to data that's in geographic coordinates
    dsm_higher_resolution = np.mean(np.abs(dsm.rio.resolution())) < np.mean(
        np.abs(dtm_resampled.rio.resolution())
    )
    # If the DSM is higher resolution it will be coarstened, so use averaging. Else use bicubic.
    resampling = Resampling.average if dsm_higher_resolution else Resampling.cubic
    # Pixel-align the dsm to the DTM
    dsm_resampled = dsm.rio.reproject_match(dtm_resampled, resampling=resampling)

    # Subtract the two products
    chm = dsm_resampled - dtm_resampled
    # Set all negative values to zero
    chm = chm.clip(min=0)

    # Save to disk
    Path(output_chm_path).parent.mkdir(parents=True, exist_ok=True)
    chm.rio.to_raster(output_chm_path)

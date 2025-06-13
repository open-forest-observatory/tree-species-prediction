# %%
import time

import matplotlib.pyplot as plt
from tree_detection_framework.detection.detector import GeometricDetector
from tree_detection_framework.postprocessing.postprocessing import (
    merge_and_postprocess_detections,
    multi_region_hole_suppression,
    multi_region_NMS,
    single_region_hole_suppression,
)
from tree_detection_framework.preprocessing.preprocessing import (
    create_dataloader,
    visualize_dataloader,
)


def detect_tree(
    raster_file,
    save_path,
    verbose=False,
):
    dataloader = create_dataloader(
        raster_folder_path=raster_file,
        chip_size=512,
        chip_stride=400,
        batch_size=3,
        output_resolution=0.2,
    )

    detector = GeometricDetector(
        a=0,
        b=0.11,
        c=0,
        res=dataloader.sampler.res,
        confidence_factor="distance",
        filter_shape="square",
        compute_tree_crown=True,
    )

    predicted_crowns = detector.predict(dataloader)

    NMS_crowns = multi_region_NMS(predicted_crowns)

    NMS_crowns.save(save_path=save_path)


if __name__ == "__main__":
    detect_tree(
        "/ofo-share/repos-david/tree-species-prediction/scratch/chm_average.tif",
        "/ofo-share/repos-david/tree-species-prediction/scratch/detected_trees.gpkg",
        verbose=True,
    )

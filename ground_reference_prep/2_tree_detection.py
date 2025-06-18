# %%
from pathlib import Path
from tree_detection_framework.preprocessing.preprocessing import (
    create_dataloader,
    create_intersection_dataloader,
)
from tree_detection_framework.detection.detector import (
    GeometricTreeTopDetector,
    GeometricTreeCrownDetector,
)
from tree_detection_framework.postprocessing.postprocessing import (
    remove_edge_detections,
    multi_region_NMS,
)

RASTER_FOLDER_PATH = "/ofo-share/cv-itd-eval_data/photogrammetry-outputs/emerald-point_10a-20230103T2008/chm.tif"
CHIP_SIZE = 1024
CHIP_STRIDE = 800
OUTPUT_RESOLUTION = 0.2


def detect_trees(
    raster_file,
    save_folder,
    chip_size=CHIP_SIZE,
    chip_stride=CHIP_STRIDE,
    output_resolution=OUTPUT_RESOLUTION,
):
    # Stage 1: Create a dataloader for the raster data and detect the tree-tops
    dataloader = create_dataloader(
        raster_folder_path=raster_file,
        chip_size=chip_size,
        chip_stride=chip_stride,
        output_resolution=output_resolution,
    )

    treetop_detector = GeometricTreeTopDetector(
        a=0, b=0.11, c=0, res=output_resolution, confidence_feature="distance"
    )

    treetop_detections = treetop_detector.predict(dataloader)

    # TODO some sort of NMS

    treetop_detections = remove_edge_detections(
        treetop_detections,
        suppression_distance=(chip_size - chip_stride) * output_resolution / 2,
    )

    treetop_detections.save(Path(save_folder, "tree_tops.gpkg"))

    # Stage 2: Combine raster and vector data (from the tree-top detector) to create a new dataloader
    raster_vector_dataloader = create_intersection_dataloader(
        raster_data=raster_file,
        vector_data=treetop_detections,
        chip_size=chip_size,
        chip_stride=chip_stride,
        output_resolution=output_resolution,
    )

    treecrown_detector = GeometricTreeCrownDetector(
        res=output_resolution, confidence_feature="distance"
    )

    treecrown_detections = treecrown_detector.predict(raster_vector_dataloader)
    treecrown_detections = multi_region_NMS(
        treecrown_detections, confidence_column="score", intersection_method="IOS"
    )
    treecrown_detections.save(Path(save_folder, "tree_crowns.gpkg"))


if __name__ == "__main__":
    detect_trees(
        "/ofo-share/repos-david/tree-species-prediction/scratch/chm-mesh_121.tif",
        "/ofo-share/repos-david/tree-species-prediction/scratch/detected_trees",
    )

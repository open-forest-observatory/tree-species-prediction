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

CHIP_SIZE = 512
CHIP_STRIDE = 400
RESOLUTION = 0.2


def detect_trees(
    CHM_file: Path,
    save_folder: Path,
    chip_size: int = CHIP_SIZE,
    chip_stride: int = CHIP_STRIDE,
    resolution: float = RESOLUTION,
):
    """Detect trees geometrically and save the detected tree tops and tree crowns.

    Args:
        CHM_file (Path):
            Path to a CHM file to detect trees from
        save_folder (Path):
            Where to save the detected tree tops and crowns. Will be created if it doesn't exist.
        chip_size (int, optional):
            The size of the chip in pixels. Defaults to CHIP_SIZE.
        chip_stride (int, optional):
            The stride of the sliding chip window in pixels. Defaults to CHIP_STRIDE.
        output_resolution (float, optional):
            The spatial resolution that the CHM is resampled to. Defaults to OUTPUT_RESOLUTION.
    """
    # Stage 1: Create a dataloader for the raster data and detect the tree-tops
    dataloader = create_dataloader(
        raster_folder_path=CHM_file,
        chip_size=chip_size,
        chip_stride=chip_stride,
        resolution=resolution,
    )

    # Create the detector for variable window maximum detection
    treetop_detector = GeometricTreeTopDetector(
        a=0, b=0.11, c=0, confidence_feature="distance"
    )

    # Generate tree top predictions
    treetop_detections = treetop_detector.predict(dataloader)

    # Remove the tree tops that were generated in the edges of tiles. This is an alternative to NMS.
    treetop_detections = remove_edge_detections(
        treetop_detections,
        suppression_distance=(chip_size - chip_stride) * resolution / 2,
    )

    treetop_detections.save(Path(save_folder, "tree_tops.gpkg"))

    # Stage 2: Combine raster and vector data (from the tree-top detector) to create a new dataloader
    raster_vector_dataloader = create_intersection_dataloader(
        raster_data=CHM_file,
        vector_data=treetop_detections,
        chip_size=chip_size,
        chip_stride=chip_stride,
        resolution=resolution,
    )

    # Create the crown detector, which is seeded by the tree top points detected in the last step
    # The score metric is how far from the edge the detection is, which prioritizes central detections
    treecrown_detector = GeometricTreeCrownDetector(confidence_feature="distance")

    # Predict the crowns
    treecrown_detections = treecrown_detector.predict(raster_vector_dataloader)
    # Suppress overlapping crown predictions. This step can be slow.
    treecrown_detections = multi_region_NMS(
        treecrown_detections, confidence_column="score", intersection_method="IOS"
    )
    # Save
    treecrown_detections.save(Path(save_folder, "tree_crowns.gpkg"))


if __name__ == "__main__":
    detect_trees(
        "/ofo-share/repos-david/tree-species-prediction/scratch/chm-mesh_121.tif",
        "/ofo-share/repos-david/tree-species-prediction/scratch/detected_trees",
    )

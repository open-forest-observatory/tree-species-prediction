import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import tifffile as tif
import os
from tqdm import tqdm
import json
import uuid
from scipy.ndimage import binary_dilation, label

import _bootstrap
from configs.path_config import path_config

# Background masking configuration
MASK_BACKGROUND = True  # Whether to mask out background trees
MASK_BUFFER_PIXELS = 20  # Buffer zone around mask in pixels to retain
BACKGROUND_VALUE = (128, 128, 128)  # Value to set background pixels (0-255, recommend 128 for mid-gray)

# Other parameters
BBOX_PADDING_RATIO = 0.02           
IMAGE_RES_CONSTRAINT = 250  # min edge length (height or width) to save
LABELLED_ONLY = True  # if using on all images (labelled and not labelled) -> set to False

# Path configurations
tree_label_mask_paths = path_config.rendered_instance_ids
raw_imgs_path = path_config.paired_image_sets_for_photogrammetry
ground_data_path = path_config.drone_crowns_with_field_attributes
output_path = path_config.cropped_tree_training_images
species_crosswalk_path = path_config.species_class_crosswalk_file

def load_all_species_mappings(crosswalk_path):
    """
    Load species crosswalk and create mapping dictionaries for all levels.
    
    Args:
        crosswalk_path (str): Path to the crosswalk CSV file
    
    Returns:
        dict: Dictionary with keys 'l1', 'l2', 'l3', 'l4', each containing 
              a mapping from original species_code to that level's species code
    """
    # Load crosswalk
    crosswalk_df = pd.read_csv(crosswalk_path)
    
    all_mappings = {}
    
    for level in ["l1", "l2", "l3", "l4"]:
        primary_col = f"primary_species_{level}"
        species_col = f"species_code_{level}"
        
        # Create mapping dictionary: species_code -> target level species (only for primary_species_lX == True)
        level_mapping = {}
        for _, row in crosswalk_df.iterrows():
            if row[primary_col]:  # Only include species where primary_species_lX is True
                level_mapping[row['species_code']] = row[species_col]
        
        all_mappings[level] = level_mapping
        print(f"Loaded {level.upper()} mapping with {len(level_mapping)} valid species")
        print(f"Available {level.upper()} classes: {sorted(set(level_mapping.values()))}")
    
    return all_mappings

def map_species_all_levels(original_species, all_mappings):
    """
    Map original species to all available levels.
    
    Args:
        original_species (str): Original species code
        all_mappings (dict): Dictionary containing mappings for all levels
    
    Returns:
        dict: Dictionary with keys 'l1', 'l2', 'l3', 'l4' containing mapped species 
              (or None if no mapping available for that level)
    """
    if original_species is None:
        return {level: None for level in ["l1", "l2", "l3", "l4"]}
    
    mapped_species = {}
    for level in ["l1", "l2", "l3", "l4"]:
        if original_species in all_mappings[level]:
            # all_mappings[level] maps original species code to the code for this level
            mapped_species[level] = all_mappings[level][original_species]
        else:
            mapped_species[level] = None
    
    return mapped_species


def filter_contours_by_area(binary_mask, area_threshold=0.5):
    """
    Filter contours in a binary mask, keeping only the largest contour and 
    any contours with area >= area_threshold * largest_contour_area.
    """
    # Label connected regions (contours) in the binary mask
    labeled_mask, num_features = label(binary_mask)

    # If only one contour, return the original mask
    if num_features <= 1:
        return binary_mask

    # Calculate area (pixel count) for each contour
    # TODO: Find a more efficient way to compute areas
    contour_areas = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    # Find largest contour area and set minimum area threshold
    largest_area = contour_areas[0][1]
    min_area = largest_area * area_threshold

    # Build mask with only contours meeting the area threshold (ideally should be only one)
    filtered_mask = np.zeros_like(binary_mask, dtype=bool)
    for contour_id, area in contour_areas:
        if area >= min_area:
            # Set elements to True wherever either filtered_mask is already True or (labeled_mask == contour_id)
            filtered_mask |= (labeled_mask == contour_id)

    return filtered_mask


def create_masked_image(img_array, tree_mask, buffer_pixels, background_value):
    """
    Create a masked version of the image where background is set to a neutral value.
   
    Args:
        img_array (np.ndarray): Original image as numpy array (H, W, C)
        tree_mask (np.ndarray): Binary mask for the tree (H, W), True for tree pixels
        buffer_pixels (int): Number of pixels to dilate the mask (buffer zone)
        background_value (tuple): Values for 3 channels to set background pixels (0-255)
   
    Returns:
        np.ndarray: Masked image array
    """
    # Create dilated mask (adds buffer around the tree)
    if buffer_pixels > 0:
        dilated_mask = binary_dilation(tree_mask, iterations=buffer_pixels)
    else:
        dilated_mask = tree_mask
   
    # Create output image
    masked_img = img_array.copy()
   
    # Set background pixels to background_value
    masked_img[~dilated_mask] = background_value
   
    return masked_img


# Load all species mappings
all_species_mappings = load_all_species_mappings(species_crosswalk_path)


# Specify datasets to process (set to None to process all datasets)
DATASETS_TO_PROCESS = [
    "0073_000874_000932",
    "0074_000874_000932",
    "0076_000874_000932",
    "0077_000810_000808",
    "0078_000810_000808",
    "0079_000810_000808",
    "0080_000810_000808"
]

if DATASETS_TO_PROCESS is None:
    dset_names = sorted(os.listdir(tree_label_mask_paths))
    print(f"Processing all {len(dset_names)} datasets")
else:
    all_dset_names = sorted(os.listdir(tree_label_mask_paths))
    dset_names = [d for d in all_dset_names if d in DATASETS_TO_PROCESS]
    print(f"Processing {len(dset_names)} specific datasets: {dset_names}")

dset_gt_mapping = {dset_name: gpd.read_file(Path(ground_data_path, dset_name+'.gpkg')) for dset_name in dset_names}

# Process each dataset individually
missing_img_ctr = 0
skipped_datasets = []
mapping_stats = {
    'total_processed': 0,
    'saved_crops': 0,
    'with_species_label': 0,
    'without_species_label': 0
}

for dset_name in dset_names:
    # metadata records for this specific dataset
    records = []
    
    dset_idx, nadir_id, oblique_id = dset_name.split("_")

    # If metadata file for this dataset already exists, skip processing it again
    labelled_output_dir = Path(output_path, "labelled", dset_name)
    metadata_fp = labelled_output_dir / f"{dset_name}_metadata.csv"
    if metadata_fp.exists():
        print(f"Skipping dataset {dset_name}: has already been processed")
        continue

    # get tif mask files from all the subdirs
    nadir_base_path = Path(tree_label_mask_paths, dset_name, 'nadir', nadir_id)
    oblique_base_path = Path(tree_label_mask_paths, dset_name, 'oblique', oblique_id)

    # Check if both folders exist
    if not nadir_base_path.exists() or not oblique_base_path.exists():
        print(f"Skipping dataset {dset_name}: missing folder(s)")
        skipped_datasets.append(dset_name)
        continue

    # assemble list of tuples for this dataset's data paths
    # each item in the list will be (src_info, mask_fp, img_path)
    # src info is a dict containing the dataset name (e.g. 0001_001435_001436), and flight_type (nadir or oblique)
    # and the mask and img paths are paired such that the mask corresponds to the image
    data_paths = []

    for flight_dir, flight_type in [(nadir_base_path, 'nadir'), (oblique_base_path, 'oblique')]:
        try:
            for mask_fp in flight_dir.rglob("*.tif"):
                # take the tif mask file path and change parts to get the corresponding image path
                rel_path = mask_fp.relative_to(tree_label_mask_paths)
                corr_img_path = raw_imgs_path / rel_path.with_suffix('.JPG')

                if mask_fp.is_file() and corr_img_path.is_file():
                    src_info = {
                        'dset_name': dset_name,
                        'flight_type': flight_type,
                        'IDs_to_labels_path': nadir_base_path.parents[1] / "IDs_to_labels.json"
                    }
                    data_paths.append((src_info, mask_fp, corr_img_path))
                else:
                    print(f"WARNING: mask file: {mask_fp} has no corresponding image file")
                    missing_img_ctr += 1
        except FileNotFoundError as fne:
            print(f"WARNING: Missing label directory {flight_dir}")
            continue

    if not data_paths:
        print(f"No data paths found for dataset {dset_name}")
        continue

    pbar = tqdm(data_paths, unit="file", position=0, leave=True, dynamic_ncols=True)
    pbar.set_description(f"Processing dataset: {dset_name}")
    
    for src_info, mask_file_path, img_file_path in pbar:
        plot_attributes = dset_gt_mapping[src_info['dset_name']][['unique_ID', 'species_code']] # id and species cols of ground ref geodataframe
        labelled_tree_ids = plot_attributes[plot_attributes.species_code.notnull()].unique_ID # get trees with species label
        labelled_tree_ids = labelled_tree_ids.to_numpy(int)

        img = Image.open(img_file_path) # load image
        img_array = np.array(img) if MASK_BACKGROUND else None  # Convert to numpy array for masking
        img_cx, img_cy = img.width / 2, img.height / 2
        safe_radius = min(img_cx, img_cy) # how far a crop can be from the center to be acceptable distortion
        mask_ids = tif.imread(mask_file_path) # load tif tree id mask
        mask_ids = np.squeeze(mask_ids) # (H, W, 1) -> (H, W)
        unique_mask_values = np.unique(mask_ids) # get unique mask ids

        # load the json file that maps the mask IDs to their unique_ID values and create a mapping dict
        IDs_to_labels_path = src_info["IDs_to_labels_path"]
        with open(IDs_to_labels_path, "r") as f:
            IDs_to_labels = json.load(f)
        ID_mapping = {int(k): int(v) for k, v in IDs_to_labels.items()}

        if LABELLED_ONLY:
            # filter mask values to only those that map to labelled IDs
            unique_mask_values = [uid for uid in unique_mask_values 
                                  if ID_mapping.get(int(uid)) in labelled_tree_ids]

        # iterate over ids
        for tree_mask_value in unique_mask_values:
            if np.isnan(tree_mask_value): # skip nan values
                continue

            # map mask value -> tree unique ID
            tree_unique_id = ID_mapping.get(int(tree_mask_value))
            if tree_unique_id is None:
                raise ValueError(f"Tree unique_ID for mask value {tree_mask_value} is None. Check {IDs_to_labels_path}")

            tree_unique_id = str(int(tree_unique_id)).zfill(5) # convert tree uid to 0 padded str to match plot_attributes format

            # see if this tree has a species label
            species_val = plot_attributes.loc[plot_attributes['unique_ID'] == tree_unique_id, 'species_code']
            if len(species_val) == 0:
                original_species = None
            else:
                original_species = species_val.iloc[0]
                if pd.isna(original_species):
                    original_species = None

            mapping_stats['total_processed'] += 1

            # Map species to all levels
            mapped_species_all_levels = map_species_all_levels(original_species, all_species_mappings)
            
            if original_species is not None:
                mapping_stats['with_species_label'] += 1
            else:
                mapping_stats['without_species_label'] += 1

            # Determine directory structure based on whether species exists
            species_dir_label = 'unlabelled' if original_species is None else 'labelled'

            # Create binary mask for current tree and filter contours
            tree_binary_mask = (mask_ids == tree_mask_value)
            filtered_mask = filter_contours_by_area(tree_binary_mask, area_threshold=0.5)
            
            ys, xs = np.where(filtered_mask)
            
            if ys.size == 0 or xs.size == 0:
                continue  # skip if no pixels found after filtering

            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max() # current tree img bounds

            # check for 'bad' crops
            # currently looks for low res images (h or w < 250 px),
            # or crops near the edge of the images liable for distortion (any bbox edge <250px away from full img edge)

            # low res img
            # typically trees too far off in the distance
            if y1 - y0 < IMAGE_RES_CONSTRAINT or x1 - x0 < IMAGE_RES_CONSTRAINT:
                continue

            # check if image within safe radius to limit distortion effects
            crop_cx, crop_cy = (x0 + x1) / 2, (y0 + y1) / 2
            diff = np.array([img_cx - crop_cx, img_cy - crop_cy])
            if diff @ diff > safe_radius ** 2: # similar to euclidean norm but without sqrt -> faster
                continue

            # cropped near edge
            # near edge may not capture whole tree
            # radial detection above should capture most of these but this exists as a failsafe
            if img.height - y1 < IMAGE_RES_CONSTRAINT or img.width - x1 < IMAGE_RES_CONSTRAINT \
            or y0 < IMAGE_RES_CONSTRAINT or x0 < IMAGE_RES_CONSTRAINT:
                continue

            # Apply background masking if enabled
            if MASK_BACKGROUND:
                masked_img_array = create_masked_image(
                    img_array,
                    filtered_mask,
                    MASK_BUFFER_PIXELS,
                    BACKGROUND_VALUE
                )
               
                # Convert back to PIL Image for cropping
                img_to_crop = Image.fromarray(masked_img_array)
            else:
                img_to_crop = img

            # crop image to bbox plus padding
            # cropping in PIL is left, top, right, bottom
            bbox_pad_width, bbox_pad_height = (x1 - x0)  * BBOX_PADDING_RATIO, (y1 - y0) * BBOX_PADDING_RATIO
            bbox = (x0 - bbox_pad_width, y0 - bbox_pad_height, x1 + bbox_pad_width, y1 + bbox_pad_height)
            cropped_img = img_to_crop.crop(bbox)

            fp = Path(
                output_path,                        # base dir to the training data folder
                species_dir_label,                  # separate trees with species labels or no species labels
                src_info['dset_name'],              # original dataset name (src folder name of images/masks)
                f"treeID{tree_unique_id}"           # tree id from mask
            ).resolve()
            os.makedirs(fp, exist_ok=True)

            # generate random 10-digit ID for the image
            image_id = str(uuid.uuid4().int)[:10]   # take first 10 digits of a UUID-derived int
            crop_path = fp / f"{image_id}.png"

            cropped_img.save(crop_path) # save cropped img
            mapping_stats['saved_crops'] += 1

            # Build metadata record with all species levels
            record = {
                "image_id": image_id,
                "image_path": str(crop_path.resolve()),
                "mask_value": tree_mask_value,
                "tree_unique_id": tree_unique_id,
                "species_original": original_species,
                "species_l1": mapped_species_all_levels['l1'],
                "species_l2": mapped_species_all_levels['l2'],
                "species_l3": mapped_species_all_levels['l3'],
                "species_l4": mapped_species_all_levels['l4'],
                "dataset_name": src_info['dset_name'],
                "flight_type": src_info['flight_type'],
                "source_image_path": str(img_file_path),
            }

            records.append(record)

    # save per-dataset metadata CSV after processing each dataset
    if records:
        columns = [
            "image_id", "image_path", "mask_value", "tree_unique_id", 
            "species_original", "species_l1", "species_l2", "species_l3", "species_l4",
            "dataset_name", "flight_type", "source_image_path"
        ]
        
        meta_df = pd.DataFrame(records, columns=columns)
        
        metadata_fp = labelled_output_dir / f"{dset_name}_metadata.csv"
        meta_df.to_csv(metadata_fp, index=False)
        print(f"Metadata saved to {metadata_fp}")
    else:
        print(f"No records to save for dataset {dset_name}")

# Print final summary
print(f"Processing complete!")
print(f"Found {missing_img_ctr} mask files without corresponding image files") # should be 0
if skipped_datasets:
    print(f"Skipped datasets due to missing folders: {skipped_datasets}")

print(f"Summary:")
print(f"Total trees processed: {mapping_stats['total_processed']}")
print(f"Trees with species labels: {mapping_stats['with_species_label']}")
print(f"Trees without species labels: {mapping_stats['without_species_label']}")
print(f"Total crops saved: {mapping_stats['saved_crops']}")

# Print summary of available classes at each level
for level in ["l1", "l2", "l3", "l4"]:
    classes = sorted(set(all_species_mappings[level].values()))
    print(f"\nAvailable {level.upper()} classes ({len(classes)}): {classes}")
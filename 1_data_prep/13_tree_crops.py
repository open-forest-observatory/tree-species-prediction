import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import tifffile as tif
import os
from tqdm import tqdm
import json
import uuid   # for generating random IDs

import _bootstrap
from configs.path_config import path_config

# Species mapping configuration
MAPPING_LEVEL = "l1"  # Options: "l1", "l2", "l3", "l4", or None for no mapping
INCLUDE_UNMAPPED_SPECIES = True  # Whether to save crops for species not in mapping

# Other parameters
BBOX_PADDING_RATIO = 0.02           
IMAGE_RES_CONSTRAINT = 250  # min edge length (height or width) to save
LABELLED_ONLY = True  # if using on all images (labelled and not labelled) -> set to False

# Path configurations
tree_label_mask_paths = path_config.rendered_instance_ids
raw_imgs_path = path_config.raw_image_sets_folder
ground_data_path = path_config.drone_crowns_with_field_attributes
output_path = path_config.cropped_tree_training_images
species_crosswalk_path = path_config.species_class_crosswalk_file

def load_species_mapping(crosswalk_path, level):
    """
    Load species crosswalk and create mapping dictionary for specified level.
    
    Args:
        crosswalk_path (str): Path to the crosswalk CSV file
        level (str): Mapping level ("l1", "l2", "l3", "l4") or None for no mapping
    
    Returns:
        dict: Mapping from original species_code to target level species code
        None: If level is None (no mapping)
    """
    if level is None:
        print("No species mapping specified - using original species codes")
        return None
    
    if level not in ["l1", "l2", "l3", "l4"]:
        raise ValueError(f"Invalid mapping level '{level}'. Must be one of: l1, l2, l3, l4, or None")
    
    # Load crosswalk
    crosswalk_df = pd.read_csv(crosswalk_path)
    
    # Column names for the specified level
    primary_col = f"primary_species_{level}"
    species_col = f"species_code_{level}"
    
    # Create mapping dictionary: species_code -> target level species (only for primary_species_lX == True)
    species_mapping = {}
    for _, row in crosswalk_df.iterrows():
        if row[primary_col]:  # Only include species where primary_species_lX is True
            species_mapping[row['species_code']] = row[species_col]
    
    print(f"Loaded species crosswalk for level {level.upper()} with {len(species_mapping)} valid mappings")
    print(f"Available {level.upper()} classes: {sorted(set(species_mapping.values()))}")
    
    return species_mapping

def map_species(original_species, species_mapping, level):
    """
    Map original species to target level, handling unmapped species.
    
    Args:
        original_species (str): Original species code
        species_mapping (dict): Mapping dictionary (None if no mapping)
        level (str): Target mapping level
    
    Returns:
        tuple: (mapped_species, is_mapped)
            - mapped_species: Target level species or original species if no mapping
            - is_mapped: Boolean indicating if mapping was successful
    """
    if species_mapping is None:
        return original_species, True  # No mapping requested, consider all as "mapped"
    
    if original_species in species_mapping:
        return species_mapping[original_species], True
    else:
        return original_species, False

# Load species mapping based on configuration
species_mapping = load_species_mapping(species_crosswalk_path, MAPPING_LEVEL)

# assemble list of dirs
# ["0001_001435_001436", "0002_000451_000446", ...]
dset_names = sorted(os.listdir(tree_label_mask_paths))
dset_gt_mapping = {dset_name: gpd.read_file(Path(ground_data_path, dset_name+'.gpkg')) for dset_name in dset_names}

# Process each dataset individually
missing_img_ctr = 0
skipped_datasets = []
mapping_stats = {
    'total_processed': 0,
    'mapped_species': 0,
    'unmapped_species': 0,
    'saved_crops': 0,
    'skipped_unmapped': 0
}

for dset_name in dset_names:
    # metadata records for this specific dataset
    records = []
    
    dset_idx, nadir_id, oblique_id = dset_name.split("_")

    # Check if output folder for this dataset already exists in 'labelled'
    labelled_output_dir = Path(output_path, "labelled", dset_name)
    if labelled_output_dir.exists():
        print(f"Skipping dataset {dset_name}: output folder already exists in 'labelled'")
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

            # Handle species mapping
            if original_species is not None:
                mapped_species, is_mapped = map_species(original_species, species_mapping, MAPPING_LEVEL)
                
                if is_mapped:
                    mapping_stats['mapped_species'] += 1
                    final_species = mapped_species
                else:
                    mapping_stats['unmapped_species'] += 1
                    if INCLUDE_UNMAPPED_SPECIES:
                        final_species = original_species  # Keep original species
                    else:
                        mapping_stats['skipped_unmapped'] += 1
                        continue  # Skip this tree entirely
            else:
                # No species label
                final_species = None
                is_mapped = False

            # Determine file naming and directory structure
            species_file_label = 'NONE' if final_species is None else str(final_species)
            species_dir_label = 'unlabelled' if final_species is None else 'labelled'

            ys, xs = np.where(mask_ids == tree_mask_value) # gather all pixels of current tree
            if ys.size == 0 or xs.size == 0:
                continue  # skip if no pixels found

            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max() # current tree img bounds

            # check for 'bad' crops
            # currently looks for low res images (h or w < 200 px), 
            # or crops near the edge of the images liable for distortion (any bbox edge <200px away from full img edge)
            # TODO: also try laplacian variance and/or tenengrad score, as well as checking for low contrast

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

            # crop image to bbox plus padding
            # cropping in PIL is left, top, right, bottom
            bbox_pad_width, bbox_pad_height = (x1 - x0)  * BBOX_PADDING_RATIO, (y1 - y0) * BBOX_PADDING_RATIO
            bbox = (x0 - bbox_pad_width, y0 - bbox_pad_height, x1 + bbox_pad_width, y1 + bbox_pad_height)
            cropped_img = img.crop(bbox)

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

            # Build metadata record with dynamic columns based on mapping level
            record = {
                "image_id": image_id,
                "image_path": str(crop_path.resolve()),  # full path to file
                "mask_value": tree_mask_value,              # original mask ID
                "tree_unique_id": tree_unique_id,        # unique ID from ground data
                "species_original": original_species,   # original species code
                "dataset_name": src_info['dset_name'],
                "flight_type": src_info['flight_type'],
                "source_image_path": img_file_path,    # path to original full image
                "mapping_level": MAPPING_LEVEL if MAPPING_LEVEL else "none",
                "is_mapped": is_mapped if original_species is not None else None,
            }
            
            # Add level-specific columns
            if MAPPING_LEVEL:
                record[f"species_{MAPPING_LEVEL}"] = final_species if is_mapped else None
            
            # Final species column (what's actually used for file naming)
            record["species_final"] = species_file_label

            records.append(record)

    # save per-dataset metadata CSV after processing each dataset
    if records:
        # Build column list
        base_columns = [
            "image_id", "image_path", "mask_value", "tree_unique_id", 
            "species_original"
        ]
        
        if MAPPING_LEVEL:
            base_columns.append(f"species_{MAPPING_LEVEL}")
        
        base_columns.extend([
            "species_final", "dataset_name", "flight_type", 
            "mapping_level", "is_mapped"
        ])
        
        meta_df = pd.DataFrame(records, columns=base_columns)
        
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

print(f"Species mapping summary:")
print(f"Mapping level: {MAPPING_LEVEL if MAPPING_LEVEL else 'None (original species)'}")
print(f"Include unmapped species: {INCLUDE_UNMAPPED_SPECIES}")
print(f"Total trees processed: {mapping_stats['total_processed']}")
print(f"Successfully mapped species: {mapping_stats['mapped_species']}")
print(f"Unmapped species: {mapping_stats['unmapped_species']}")
print(f"Total crops saved: {mapping_stats['saved_crops']}")
if not INCLUDE_UNMAPPED_SPECIES:
    print(f"Crops skipped (unmapped): {mapping_stats['skipped_unmapped']}")

if species_mapping:
    print(f"Available {MAPPING_LEVEL.upper()} classes: {sorted(set(species_mapping.values()))}")
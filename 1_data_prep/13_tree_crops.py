import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import sys
import tifffile as tif
import os
from tqdm import tqdm
import itertools
import json

import time

import _bootstrap
from configs.path_config import path_config

# pad bbox of cropped images
# actual padding is this ratio times the edge len
BBOX_PADDING_RATIO = 0.02           
IMAGE_RES_CONSTRAINT = 250  # min edge length (height or width) to save

# if using on all images (labelled and not labelled) -> set to False
LABELLED_ONLY = True

# rendered instance IDs (.tif) files
tree_label_mask_paths = path_config.rendered_instance_ids

# raw images folder
raw_imgs_path = path_config.raw_image_sets_folder

# ground data information (.gpkg) files
ground_data_path = path_config.drone_crowns_with_field_attributes

# output to save cropped images
output_path = path_config.cropped_tree_training_images

''' sample file paths for reference
dset1_name = "0001_001435_001436"
nadir1 = Path("nadir/001435/001435-01/00")
oblique1 = Path("oblique/001436/001436-01/00")

dset2_name = "0002_000451_000446"
nadir2 = Path("nadir/000451/000451-01/00")
oblique2 = Path("oblique/000446/000446-01/00")
'''

# assemble list of dirs
# ["0001_001435_001436", "0002_000451_000446", ...]
dset_names = sorted(os.listdir(tree_label_mask_paths))
gdfs = {dset_name: gpd.read_file(Path(ground_data_path, dset_name+'.gpkg')) for dset_name in dset_names}

# assemble list of tuples for the data paths
# each item in the list will be (src_info, mask_fp, img_path)
# src info is a dict containing the dataset name (e.g. 0001_001435_001436), and flight_type (nadir or oblique)
# and the mask and img paths are paired such that the mask corresponds to the image
data_paths = []
missing_img_ctr = 0
skipped_datasets = []

for dset_name in dset_names:
    dset_tif = dset_name + '' # originally used _npy extension, leaving here in case render label names are ever different
    dset_idx, nadir_id, oblique_id = dset_name.split("_")

    # Check if output folder for this dataset already exists in 'labelled'
    labelled_output_dir = Path(output_path, "labelled", dset_name)
    if labelled_output_dir.exists():
        print(f"Skipping dataset {dset_name}: output folder already exists in 'labelled'")
        continue

    # get tif mask files from all the subdirs
    nadir_base_path = Path(tree_label_mask_paths, dset_tif, 'nadir', nadir_id)
    oblique_base_path = Path(tree_label_mask_paths, dset_tif, 'oblique', oblique_id)

    # Check if both folders exist
    if not nadir_base_path.exists() or not oblique_base_path.exists():
        print(f"Skipping dataset {dset_name}: missing folder(s)")
        skipped_datasets.append(dset_name)
        continue

    for flight_dir, flight_type in [(nadir_base_path, 'nadir'), (oblique_base_path, 'oblique')]:
        try:
            for subdir1 in os.listdir(flight_dir):
                for subdir2 in os.listdir(flight_dir / subdir1):
                    data_path = flight_dir / subdir1 / subdir2
                    for f in os.listdir(data_path):
                        mask_fp = data_path / f # pull path of tif file

                        # take the tif mask file path and change parts to get the corresponding image path
                        rel_path = mask_fp.relative_to(tree_label_mask_paths)
                        corr_img_path = raw_imgs_path / rel_path.with_suffix('.JPG')

                        if mask_fp.is_file() and corr_img_path.is_file():
                            src_info = {
                                'dset_name': dset_name,
                                'flight_type': flight_type,
                                'IDs_to_labels_path': nadir_base_path.parent.parent / "IDs_to_labels.json"
                            }
                            data_paths.append((src_info, mask_fp, corr_img_path))
                        else:
                            print(f"WARNING: mask file: {mask_fp} has no corresponding image file")
                            missing_img_ctr += 1
        except FileNotFoundError as fne:
            print(f"WARNING: Missing label directory {flight_dir}")
            continue

# should be 0
print(f"Found {missing_img_ctr} mask files without corresponding image files")
if skipped_datasets:
    print(f"Skipped datasets due to missing folders: {skipped_datasets}")

pbar = tqdm(data_paths, unit="file", position=0, leave=True, dynamic_ncols=True)
for src_info, mask_file_path, img_file_path in pbar:
    pbar.set_description(f"Cur plot: {src_info['dset_name']}")
    gdf = gdfs[src_info['dset_name']][['unique_ID', 'species_code']] # id and species cols of ground ref geodataframe
    labelled_tree_ids = gdf[gdf.species_code.notnull()].unique_ID # get trees with species label
    labelled_tree_ids = labelled_tree_ids.to_numpy(int)

    img = Image.open(img_file_path) # load image
    img_cx, img_cy = img.width / 2, img.height / 2
    safe_radius = min(img_cx, img_cy) # how far a crop can be from the center to be acceptable distortion
    #mask_ids = np.load(mask_file_path) # load npy tree id mask
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
            continue

        tree_unique_id = str(int(tree_unique_id)).zfill(5) # convert tree uid to 0 padded str to match gdf format

        # see if this tree has a species label
        species_val = gdf.loc[gdf['unique_ID'] == tree_unique_id, 'species_code']
        if len(species_val) == 0:
            species_code = None
        else:
            species_code = species_val.iloc[0]
            if pd.isna(species_code):
                species_code = None

        species_file_label = 'NONE' if species_code is None else str(species_code)
        species_dir_label = 'unlabelled' if species_code is None else 'labelled'

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

        #output_path = "2_training/data/cropped_trees/"
        fp = Path(
            output_path,                        # base dir to the training data folder
            species_dir_label,              # separate trees with species labels or no species labels
            src_info['dset_name'],          # original dataset name (src folder name of images/masks)
            f"treeID{tree_unique_id}"          # tree id from mask
        ).resolve()
        os.makedirs(fp, exist_ok=True)

        # cropped img file name
        fn = f"treeID{tree_unique_id}-species{species_file_label}-dset{src_info['dset_name']}-view{src_info['flight_type']}.png"
        crop_path = fp / fn

        # add a ctr to not overwrite other saved crops of same trees
        for i in itertools.count(1): # start at 2 since first image gets a 1
            candidate = crop_path.with_name(f"{crop_path.stem}-{i}{crop_path.suffix}")
            if not candidate.exists():
                crop_path = candidate
                break

        cropped_img.save(crop_path) # save cropped img
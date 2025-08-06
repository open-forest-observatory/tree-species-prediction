import numpy as np
import geopandas as gpd
from pathlib import Path
from PIL import Image
import sys
import tifffile as tif
import os
from tqdm import tqdm
import itertools

import _bootstrap
from configs.path_config import path_config


'''TODO
only save out cropped trees with species_id
change treetop_id to unique_id
don't need to worry about grouping trees between drone missions
rerun ensure cropped images aren't overwriting
switch to tiff instead of npy
radial cropping based on min(im_w, im_h) / 2
'''

# pad bbox of cropped images
# actual padding is this ratio times the edge len
BBOX_PADDING_RATIO = 0.02           
IMAGE_RES_CONSTRAINT = 250  # min edge length (height or width) to save

# rendered instance IDs (.tif) files
tree_label_masks = path_config.rendered_instance_ids_path

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
dset_names = sorted(os.listdir(tree_label_masks))
gdfs = {dset_name: gpd.read_file(Path(ground_data_path, dset_name+'.gpkg')) for dset_name in dset_names}

# assemble list of tuples for the data paths
# each item in the list will be (src_info, mask_fp, img_path)
# src info is a dict containing the dataset name (e.g. 0001_001435_001436), and flight_type (nadir or oblique)
# and the mask and img paths are paired such that the mask corresponds to the image
data_paths = []
missing_img_ctr = 0
for dset_name in dset_names:
    dset_tif = dset_name + '' # originally used _npy extension, leaving here in case render label names are ever different
    dset_idx, nadir_id, oblique_id = dset_name.split("_")

    # get tif mask files from all the subdirs
    nadir_base_path = Path(tree_label_masks, dset_tif, 'nadir', nadir_id)
    oblique_base_path = Path(tree_label_masks, dset_tif, 'oblique', oblique_id)

    for flight_dir, flight_type in [(nadir_base_path, 'nadir'), (oblique_base_path, 'oblique')]:
        for subdir1 in os.listdir(flight_dir):
            for subdir2 in os.listdir(flight_dir / subdir1):
                data_path = flight_dir / subdir1 / subdir2
                for f in os.listdir(data_path):
                    mask_fp = data_path / f # pull path of tif file

                    # take the tif mask file path and change parts to get the corresponding image path
                    path_parts = mask_fp.parts
                    # replace mask base bath with img base path
                    split_idx = path_parts.index(tree_label_masks.parts[-1])
                    after = list(path_parts[split_idx+1:])
                    #after[0] = after[0].replace('_npy', '') # remove '_npy' in dset name for imgs
                    after[-1] = mask_fp.stem + '.JPG' # replace '.tif' file ext with img file ext
                    corr_img_path = Path(raw_imgs_path, *after) # rebuild img path

                    if mask_fp.is_file() and corr_img_path.is_file():
                        src_info = {
                            'dset_name': dset_name,
                            'flight_type': flight_type,
                        }
                        data_paths.append((src_info, mask_fp, corr_img_path))
                    else:
                        print(f"WARNING: mask file: {mask_fp} has no corresponding image file")
                        missing_img_ctr += 1

# should be 0
print(f"Found {missing_img_ctr} mask files without corersponding image files")

pbar = tqdm(data_paths, unit="file")
for src_info, mask_file_path, img_file_path in pbar:
    pbar.set_description(src_info['dset_name'])
    gdf = gdfs[src_info['dset_name']][['unique_ID', 'species_code']] # id and species cols of ground ref geodataframe
    labelled_tree_ids = gdf[gdf.species_code.notnull()].unique_ID # for now only need labelled trees
    labelled_tree_ids = labelled_tree_ids.to_numpy(int)

    img = Image.open(img_file_path) # load image
    img_cx, img_cy = img.width / 2, img.height / 2
    safe_radius = min(img_cx, img_cy) # how far a crop can be from the center to be acceptable distortion
    #mask_ids = np.load(mask_file_path) # load npy tree id mask
    mask_ids = tif.imread(mask_file_path) # load tif tree id mask
    mask_ids = np.squeeze(mask_ids) # (H, W, 1) -> (H, W)
    uids = np.unique(mask_ids) # get unique tree ids
    uids = uids[np.isin(uids, labelled_tree_ids)] # will skip unlabelled trees

    # iterate over ids
    subpbar = tqdm(uids, unit='id')
    for tree_id in subpbar:
        if np.isnan(tree_id): # skip nan values
            continue

        # see if this tree has a species label
        # NOTE: this was written to save both labelled and unlabelled,
        # but changes made above to only look for uids that have labels.
        # this is left here in case unlabelled ones needed later 
        tree_id_str = str(int(tree_id)).zfill(5) # convert mask's tree id to 0 padded str to match gdf for comparison
        try: # df is a bit odd, sometimes iloc[0] throws an error
            species_code = gdf.loc[gdf['unique_ID'] == tree_id_str, 'species_code'].iloc[0]
        except Exception as e:
            species_code = gdf.loc[gdf['unique_ID'] == tree_id_str, 'species_code']

        species_file_label = 'NONE' if species_code is None else species_code
        species_dir_label = 'unlabelled' if species_code is None else 'labelled'

        ys, xs = np.where(mask_ids == tree_id) # gather all pixels of current tree
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max() # current tree img bounds

        # check for 'bad' crops
        # currently looks for low res images (h or w < 200 px), 
        # or crops near the edge of the images liable for distortion (any bbox edge <200px away from full img edge)
        # may also try laplacian variance and/or tenengrad score

        # low res img
        # typically trees too far off in the distance
        if y1 - y0 < IMAGE_RES_CONSTRAINT or x1 - x0 < IMAGE_RES_CONSTRAINT:
            continue

        # check if image within safe radius to limit distortion effects
        crop_cx, crop_cy = (x0 + x1) / 2, (y0 + y1) / 2
        if (img_cx - crop_cx) ** 2 + (img_cy - crop_cy) ** 2 > safe_radius ** 2:
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
            f"treeID{tree_id_str}"          # tree id from mask
        ).resolve()
        os.makedirs(fp, exist_ok=True)

        # cropped img file name
        fn = f"treeID{tree_id_str}-species{species_file_label}-dset{src_info['dset_name']}-view{src_info['flight_type']}.png"
        crop_path = fp / fn

        # add a ctr to not overwrite other saved crops of same trees
        for i in itertools.count(1): # start at 2 since first image gets a 1
            candidate = crop_path.with_name(f"{crop_path.stem}-{i}{crop_path.suffix}")
            if not candidate.exists():
                crop_path = candidate
                break

        cropped_img.save(crop_path) # save cropped img
import numpy as np
import geopandas as gpd
from pathlib import Path
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

import _bootstrap
from configs.path_config import path_config

BBOX_PADDING = 10
TEST_LIMIT = 50

#rendered instance IDs (.npy) files
rendered_instance_ids_path = Path("/ofo-share/species-prediction-project/intermediate/rendered_instance_ids")

#ground data information (.gpkg) files
ground_data_path = Path("/ofo-share/species-prediction-project/intermediate/drone_crowns_with_field_attributes")

#raw images folder
raw_imgs_path = Path("/ofo-share/species-prediction-project/intermediate/raw_image_sets")

dset1_name = "0001_001435_001436"
dset1_name_npy = "0001_001435_001436_npy"
nadir1 = Path("nadir/001435/001435-01/00")
oblique1 = Path("oblique/001436/001436-01/00")

dset2_name = "0002_000451_000446"
dset2_name_npy = "0002_000451_000446_npy"
nadir2 = Path("oblique/000451/000451-01/00")
oblique2 = Path("oblique/000446/000446-01/00")

angle_types = ['nadir', 'oblique']

gdf = gpd.read_file(ground_data_path / f"{dset1_name}.gpkg")

# gather npy files as a dict where key -> filename (no ext) and value -> full path
test_data_npy_path = rendered_instance_ids_path / dset1_name_npy / nadir1
id_files = {f.stem: f for f in test_data_npy_path.iterdir() if f.is_file() and f.suffix == '.npy'}

# gather img files as a dict where key -> filename (no ext) and value -> full path
test_data_imgs_path = raw_imgs_path / dset1_name / nadir1
img_files = {f.stem: f for f in test_data_imgs_path.iterdir() if f.is_file() and f.suffix == '.JPG'}

# ensure exact pairing of matching file names
common = id_files.keys() & img_files.keys()
# list of tuples of base file name, path to id file, path to img file for each matching npy and img file
id_img_file_pairs = [(stem, id_files[stem], img_files[stem]) for stem in sorted(common)]

test_ctr = 0
for file_stem, id_file_path, img_file_path in id_img_file_pairs:
    print(file_stem)
    img = Image.open(img_file_path) # load image
    mask_ids = np.load(id_file_path) # load npy tree id mask
    mask_ids = mask_ids.squeeze(-1) # (H, W, 1) -> (H, W)
    uids = np.unique(mask_ids) # get unique tree ids

    # iterate over ids
    for tree_id in uids:
        print(tree_id)
        if not tree_id: # skip nan values
            continue

        ys, xs = np.where(mask_ids == tree_id) # gather all pixels of current tree
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max() # current tree img bounds

        # crop image to bbox plus padding
        # cropping in PIL is left, top, right, bottom
        bbox = (x0, y0, x1 + BBOX_PADDING, y1 + BBOX_PADDING)
        cropped_img = img.crop(bbox)

        # save cropped img
        fn = f"src{file_stem}-id{int(tree_id)}.png"
        #fp = f"2_training/data/cropped_trees/{angle_types[0]}"
        fp = "2_training/data/cropped_trees/"
        fp = Path(fp).resolve()
        os.makedirs(fp, exist_ok=True)

        print(fp, fn)

        cropped_img.save(fp / fn)
        test_ctr += 1
        if test_ctr >= TEST_LIMIT:
            sys.exit()
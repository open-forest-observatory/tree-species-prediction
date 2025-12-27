# Overview
The goal of this project is to generate tree species predictions for hundreds of drone collected datasets in the Open Forest Observatory [catalog](https://openforestobservatory.org/data/drone/).

# Install
These processing steps rely on functionality from several projects. Because they have incompatible dependencies, you will need to create multiple separate conda environments for various steps. You will largely follow the instructions provided in the README file of each repository. However, if you want to ensure that the code you are using from these projects exactly matches what was used to conduct these experiments, conduct the following steps. First, clone the project locally from github. Then, from within the project, run `git checkout <tag name>` where the `<tag name>` refers to a named version of the code listed in each of the following sections. Also, there is a suggested name for the conda environment for each tool in the following sections.

## [Automate Metashape](https://github.com/open-forest-observatory/automate-metashape)
This project is a wrapper around Agisoft Metashape that allows for reproducible end-to-end automated photogrammetry. The tag is `vx.x.x` and the conda environment should be called `meta`.

## [Geograypher](https://github.com/open-forest-observatory/geograypher)
This project supports indentifying the location in an image of a geospatial point and vice versa. This project requires a diverse set of dependencies, so its environment can also be used to run multiple other steps.

## [Tree Detection Framework](https://github.com/open-forest-observatory/tree-detection-framework)
This project supports multiple approaches to detecting trees in geospatial data.

# Setup
Before any scripts can be ran, you must symlink the `_bootstrap.py` script into all working directories (i.e. folders with scripts that are ran directly, not called like other scripts such as those in `utils/` or `config/`). Fortunately the script `symlink_bootstrap.py` exists for that purpose. Simply run that script from the root directory of the project and it will take care of the rest. **This only needs to be ran once, unless symlinks break, or if new working directories are added.**

**Note for Contributors:** There is a list in this script called `WORKING_DIRS` that contains the relative paths of folders described above. If you add a new folder for scripts to be ran directly, please ensure you also add the relative path of the folder to this list. Similarly if additional outside scripts are needed (e.g. Automate-Metashape) the paths to these script dirs can be added to `_bootstrap.py` in a list towards the bottom called `SCRIPTS_PATHS`.

# Data preparation
- `01_get_mission_altitude_driver_script.py`: This computes the height of the image locations above the corresponding digital surface model to compute an altitide above ground at which the image was captured. It should be run with TODO dependencies.
- `02_merge_alt_columns.py`: TODO
- `03_pair_drone_with_ground.py`: There are three important types of imput data collected in different locations: field reference surveys, drone imagery collected at a low altitude (80m) and oblique orientation and drone imagery collected at a high altitude (120m) and nadir orientation. The goal of this script is to find triples of one of each type of data that spatially overlap. Specifically, the both drone surveys should both fully cover the field plot. This should be run with TODO dependencies.
- `04_crop_raw_images_to_intersection.py`: Once the field plots and drone surveys have been assigned into triples, the images which are near the field plot can be identified. This step requires metadata which is stored in the cloud and will be downloaded automatically. So make sure to follow the instructions in [this document](https://docs.google.com/document/d/155AP0P3jkVa-yT53a-QLp7vBAfjRa78gdST1Dfb4fls/edit?usp=sharing) prior to running anything. This script can be run in the `geograypher` environment.
- `05_produce_combined_photogrammetry.py`: The images, both oblique and nadir, must be registered together using photogrammetry to complete the downstream tasks. This script creates config files which specify how to run photogrammetry, and can be run in the `meta` environment. Since photogrammetry requires a large amount of computational resources and processing each dataset is slow, we use a workflow manager to run each dataset independently in a containerized manner. This is done with the [ofo-argo](https://github.com/open-forest-observatory/ofo-argo) project. Once you have created the config files, follow the instructions in the `ofo-argo` repository to run photogtrammetry on each one. TODO, this now has other functionality built in (e.g. CHM generation and uploading to S3) that does not need to be run
- `06_compute_CHM.py`: The canopy height model (CHM) represnts the difference between the heighest point identified in the scene and the estimated ground surface. Both of these are computed by photogrammetry. This script requires [rioxarray](https://pypi.org/project/rioxarray/), [rasterio](https://rasterio.readthedocs.io/en/stable/), and [geopandas](https://geopandas.org/en/stable/) to be installed.
- `07_tree_detection.py`:
- `08_add_field_attributes_to_drone.py`
- `09_determine_species_classes.py`
- `10_render_instance_ids.py`
- `11_tree_crops.py`
- `12_create_train_val_split.py`
- `13_prepare_mmpretrain_dataset.py`



## Photogrammetry
The goal of photogrammetry is to reconstruct the 3D geometry of the scene from individual images. All steps in this section should be run with the `meta` conda environment.
- `1_produce_combined_photogrammetry.py`: Runs photogrammetry on a pair of missions, one nadir and the other oblique.

## Ground reference data prep
- `1_compute_CHM.py`: This computes a canopy height model from the digital terrain model (DTM) and digital surface model (DSM) produced by photogrammetry. It requires that [rioxarray](https://corteva.github.io/rioxarray/stable/installation.html) be installed.
- `2_tree_detection.py`: This detects trees in the CHM representation. The [Tree Detection Framework](https://github.com/open-forest-observatory/tree-detection-framework) must be installed.
- `3_match_field_drone.py`: Install [spatial-utils](https://github.com/open-forest-observatory/spatial-utils) with the exception of the `poetry install` step. Then run `conda install scikit-learn`. Finally, from within the `spatial-utils` repository, run `poetry install`.
- `09_add_field_attributes_to_drone.py`: This can be run with the tree-detection-framework dependencies. It performs matching between the field trees and drone-detected trees to add attributes from the field trees to the drone crowns. It also performs plot level filtering to remove plots with imprecisely registered reference data, low detection recall, or a high hardwood fraction. And at the tree level it removes dead trees and trees shorter than 10m.
- `13_tree_crops.py`: This can be run with the geograypher environment.

## Training
### Work in Progress
1. Download the pretrained PlantCLEF models to the appropriate directory (run these from the project root directory)
```
wget -P 2_training/models/pretrained https://zenodo.org/records/10848263/files/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar?download=1

# frozen backbone (just classification head for training)
tar -xvf 2_training/models/pretrained/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar?download=1 pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier/model_best.pth.tar --strip-components=2 -O > 2_training/models/pretrained/model_best_vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier.pth.tar

# unfrozen backbone (bigger)
tar -xvf 2_training/models/pretrained/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar?download=1 pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar --strip-components=2 -O > 2_training/models/pretrained/model_best_vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.pth.tar
```

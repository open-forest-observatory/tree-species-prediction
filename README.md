# Overview
The goal of this project is to generate tree species predictions for hundreds of drone collected datasets in the Open Forest Observatory [catalog](https://openforestobservatory.org/data/drone/).

# Install
These processing steps rely on functionality from several projects. Because they have incompatible dependencies, you will need to create multiple separate conda environments for various steps. You will largely follow the instructions provided in the README file of each repository. However, if you want to ensure that the code you are using from these projects exactly matches what was used to conduct these experiments, conduct the following steps. First, clone the project locally from github. Then, from within the project, run `git checkout <tag name>` where the `<tag name>` refers to a named version of the code listed in each of the following sections. Also, there is a suggested name for the conda environment for each tool in the following sections.

## [Automate Metashape](https://github.com/open-forest-observatory/automate-metashape)
This project is a wrapper around Agisoft Metashape that allows for reproducible end-to-end automated photogrammetry. The tag is `vx.x.x` and the conda environment should be called `meta`.

# Setup
Before any scripts can be ran, you must symlink the `_bootstrap.py` script into all working directories (i.e. folders with scripts that are ran directly, not called like other scripts such as those in `utils/` or `config/`). Fortunately the script `symlink_bootstrap.py` exists for that purpose. Simply run that script from the root directory of the project and it will take care of the rest. **This only needs to be ran once, unless symlinks break, or if new working directories are added.**

**Note for Contributors:** There is a list in this script called `WORKING_DIRS` that contains the relative paths of folders described above. If you add a new folder for scripts to be ran directly, please ensure you also add the relative path of the folder to this list. Similarly if additional outside scripts are needed (e.g. Automate-Metashape) the paths to these script dirs can be added to `_bootstrap.py` in a list towards the bottom called `SCRIPTS_PATHS`.

# Processing steps
TODO, describe overview of processing steps

## Photogrammetry
The goal of photogrammetry is to reconstruct the 3D geometry of the scene from individual images. All steps in this section should be run with the `meta` conda environment.
- `1_produce_combined_photogrammetry.py`: Runs photogrammetry on a pair of missions, one nadir and the other oblique.

## Ground reference data prep
- `1_compute_CHM.py`: This computes a canopy height model from the digital terrain model (DTM) and digital surface model (DSM) produced by photogrammetry. It requires that [rioxarray](https://corteva.github.io/rioxarray/stable/installation.html) be installed.
- `2_tree_detection.py`: This detects trees in the CHM representation. The [Tree Detection Framework](https://github.com/open-forest-observatory/tree-detection-framework) must be installed.
- `3_match_field_drone.py`: Install [spatial-utils](https://github.com/open-forest-observatory/spatial-utils) with the exception of the `poetry install` step. Then run `conda install scikit-learn`. Finally, from within the `spatial-utils` repository, run `poetry install`.
- `09_add_field_attributes_to_drone.py`: This can be run with the tree-detection-framework dependencies. It performs matching between the field trees and drone-detected trees to add attributes from the field trees to the drone crowns. It also performs plot level filtering to remove plots with imprecisely registered reference data, low detection recall, or a high hardwood fraction. And at the tree level it removes dead trees and trees shorter than 10m.

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

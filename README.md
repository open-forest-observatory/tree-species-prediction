# Overview
The goal of this project is to generate tree species predictions for hundreds of drone collected datasets in the Open Forest Observatory [catalog](https://openforestobservatory.org/data/drone/).

# Install
These processing steps rely on functionality from several projects. Because they have incompatible dependencies, you will need to create multiple separate conda environments for various steps. You will largely follow the instructions provided in the README file of each repository. However, if you want to ensure that the code you are using from these projects exactly matches what was used to conduct these experiments, conduct the following steps. First, clone the project locally from github. Then, from within the project, run `git checkout <tag name>` where the `<tag name>` refers to a named version of the code listed in each of the following sections. Also, there is a suggested name for the conda environment for each tool in the following sections.

## [Automate Metashape](https://github.com/open-forest-observatory/automate-metashape)
This project is a wrapper around Agisoft Metashape that allows for reproducible end-to-end automated photogrammetry. The tag is `vx.x.x` and the conda environment should be called `meta`.

# Processing steps
TODO, describe overview of processing steps

## Photogrammetry
The goal of photogrammetry is to reconstruct the 3D geometry of the scene from individual images. All steps in this section should be run with the `meta` conda environment.
- `1_produce_combined_photogrammetry.py`: Runs photogrammetry on a pair of missions, one nadir and the other oblique.

## Ground reference data prep
- `1_compute_CHM.py`:
- `2_tree_detection.py`:
- `3_match_field_drone.py`:

Install [spatial-utils](https://github.com/open-forest-observatory/spatial-utils)
```
conda install scikit-learn
```
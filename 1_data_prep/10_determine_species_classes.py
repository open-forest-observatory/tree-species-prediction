from pathlib import Path
import geopandas as gpd
import pandas as pd
import sys

import _bootstrap
from configs.path_config import path_config


ground_trees = gpd.read_file(path_config.ground_reference_trees_file)
ground_drone_pairs = pd.read_csv(
    path_config.ground_plot_drone_mission_matches_file,
    dtype={"plot_id": str})

# Get unique selected ground plot IDs
ground_plot_ids = ground_drone_pairs['plot_id'].unique()

# Filter ground trees to only those in the selected ground plot IDs
ground_trees = ground_trees[ground_trees['plot_id'].isin(ground_plot_ids)]


# Repeat the above two commands, but in a single pandas chain using the `agg` method
species_counts = (
    ground_trees
    .groupby('species_code')
    .agg(n_trees=('species_code', 'size'),
         n_plots=('plot_id', 'nunique'))
    .reset_index()
    .sort_values(by='n_trees', ascending=False)
)


# Prep the species classes crosswalk
species_crosswalk = species_counts


# LEVEL 1 CLASSES (only changing subspecies to species and merging the various unknown classes)

primary_species_l1 = (
    'ABCO', 'CADE27', 'NODE3', 'PSME', 'PIPO', 'PILA', 'ABMA', 'ABMAS', 'QUCH2', 'QUKE', 'PIJE',
    'PICO', 'SALIX', 'PIMO3', 'ARME', 'JUOC')
species_crosswalk['primary_species_l1'] = species_crosswalk["species_code"].isin(primary_species_l1)

mapping_l1 = {
    'ABMAS': 'ABMA',
    'UNKNCONIFER': 'UNK',
    'UNKSNAG': 'UNK',
    'UNK': 'UNK',
    'PI': 'UNK',
    'AB': 'UNK'
}
species_crosswalk['species_code_l1'] = species_crosswalk['species_code'].replace(mapping_l1)


# LEVEL 2 CLASSES (which combine some near-identical-looking species, hard for amateurs in the field)

primary_species_l2 = primary_species_l1 + ('PIPJ',)
# ^ PIPJ is a merged class of PIPO and PIJE for when they can't be distinguished in the field
species_crosswalk['primary_species_l2'] = species_crosswalk["species_code"].isin(primary_species_l2)

mapping_l2_addl = {
    'PIPO': 'PIPJ',
    'PIJE': 'PIPJ',
    'QUCH2': 'QUEV', # evergreen oak
    'QUWI2': 'QUEV', # evergreen oak   
}
mapping_l2 = mapping_l1 | mapping_l2_addl
species_crosswalk['species_code_l2'] = species_crosswalk['species_code'].replace(mapping_l2)


# LEVEL 3 CLASSES (combining species that would likely be hard for a CV model to distinguish)

primary_species_l3 = primary_species_l2 + ('AB',)
species_crosswalk['primary_species_l3'] = species_crosswalk["species_code"].isin(primary_species_l3)

mapping_l3_addl = {
    'AB': 'FIR',
    'ABCO': 'FIR',
    'ABMA': 'FIR',
    'ABMAS': 'FIR',
    'ABGR': 'FIR',
    'ABAM': 'FIR',
    'PSME': 'FIR',  # Douglas fir, not a true fir, but lookalike
    'NODE3': 'QUEV',  # evergreen oak
    'PIMO3': 'PIFIVE',
    'PILA': 'PIFIVE'
}

mapping_l3 = mapping_l2 | mapping_l3_addl
species_crosswalk['species_code_l3'] = species_crosswalk['species_code'].replace(mapping_l3)


# LEVEL 4 CLASSES (lumping to genus or family level, and all hardwoods together)

primary_species_l4 = primary_species_l3 + ('PI', 'UMCA', 'ALRU2', 'CONU4', 'ACMA3', 'POTR5', 'TSME', 'TABR2')
species_crosswalk['primary_species_l4'] = species_crosswalk["species_code"].isin(primary_species_l4)

mapping_l4_addl = {
    'CADE27': 'CUPR',
    'JUOC': 'CUPR',
    'THPL': 'CUPR',
    'PI': 'PI',
    'PIPO': 'PI',
    'PIJE': 'PI',
    'PILA': 'PI',
    'PIMO3': 'PI',
    'PIPJ': 'PI',
    'PICO': 'PI',
    'PICO3': 'PI',
    'NODE3': 'HW',
    'QUCH2': 'HW',
    'QUKE': 'HW',
    'QUWI2': 'HW',
    'QUKE': 'HW',
    'SALIX': 'HW',
    'ARME': 'HW',
    'UMCA': 'HW',
    'ALRU2': 'HW',
    'FRACAL': 'HW',
    'ARCVISM': 'HW',
    'CEAPAL': 'HW',
    'QUDE': 'HW',
    'QUEXMO': 'HW',
    'RHOOCC': 'HW',
    'CONU4': 'HW',
    'ACMA3': 'HW',
    'ACNE2': 'HW',
    'PREM': 'HW',
    'LONSUB': 'HW',
    'ACRU': 'HW',
    'POTR5': 'HW',
    'ARCPUN': 'HW',
    'ARCPRI': 'HW',
    'ARCGLA': 'HW',
    'FRPU7': 'HW',
    'FRLA': 'HW',
    'ACCI': 'HW',
    'SESE3': 'HW',
    'TSHE': "TS",
    'TSME': "TS",
    'TABR2': 'FIR',
}
mapping_l4 = mapping_l3 | mapping_l4_addl
species_crosswalk['species_code_l4'] = species_crosswalk['species_code'].replace(mapping_l4)


# Write the result
species_crosswalk.to_csv(
    path_config.species_class_crosswalk_file,
    index=False
)

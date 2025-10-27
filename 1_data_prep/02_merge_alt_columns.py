# This script is to merge the generated altitude data from individual files to the metadata file
import glob
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add folder where constants.py is to system search path
sys.path.append(str(Path(Path(__file__).parent, "..").resolve()))
from constants import (ALL_MISSIONS_METADATA_FILE,
                       DRONE_MISSIONS_WITH_ALT_FILE, MISSION_ALTITUDES_FOLDER)

metadata_gdf = gpd.read_file(ALL_MISSIONS_METADATA_FILE)

csv_files = glob.glob(os.path.join(MISSION_ALTITUDES_FOLDER, "*_altitude_summary.csv"))

altitude_data = []
for file_path in csv_files:
    filename = os.path.basename(file_path)
    match = re.match(r"(\d{6})_altitude_summary\.csv", filename)
    if match:
        mission_id = match.group(1)
        df = pd.read_csv(file_path)
        row = df.iloc[0]
        row["mission_id"] = mission_id
        altitude_data.append(row)

altitude_df = pd.DataFrame(altitude_data)
merged_gdf = metadata_gdf.merge(altitude_df, on="mission_id", how="left")  # keep all rows from left, match with right
merged_gdf.to_file(DRONE_MISSIONS_WITH_ALT_FILE)

import geopandas as gpd


def match_field_and_drone_trees(field_trees_path, drone_trees_path):
    field_trees = gpd.read_file(field_trees_path)
    drone_trees = gpd.read_file(drone_trees_path)

    # Maybe filter some of the short trees
    # Compute the full distance matrix or at least the top n matches

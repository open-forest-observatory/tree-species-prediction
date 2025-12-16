import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap
from configs.path_config import path_config

VAL_FRACTION = 0.2
SEED = 35
LUMP_LEVEL = "l3"
CROSSWALK_PATH = path_config.species_class_crosswalk_file


def load_and_apply_crosswalk(trees_gdf, crosswalk_path, lump_level="l2"):
    """
    Load species crosswalk and apply lumping to tree data.
    Only keeps trees where primary_species flag is True for the specified level.

    Args:
        trees_gdf: GeoDataFrame of individual trees
        crosswalk_path: Path to species crosswalk CSV
        lump_level: Which level to use (e.g., 'l2', 'l3', 'l4')

    Returns:
        trees_gdf with new 'lumped_species' column (filtered to primary species only)
        crosswalk dataframe for reference
    """
    crosswalk = pd.read_csv(crosswalk_path)

    # Create mapping from original species_code to lumped code
    lump_col = f"species_code_{lump_level}"
    primary_col = f"primary_species_{lump_level}"

    # Create a dictionary that maps old species code to new lumped species code
    species_mapping = dict(zip(crosswalk["species_code"], crosswalk[lump_col]))
    primary_mapping = dict(zip(crosswalk["species_code"], crosswalk[primary_col]))

    print(f"Applying species lumping at level: {lump_level}")
    print(f"Original species count: {trees_gdf['species_code'].nunique()}")
    print(f"Original tree count: {len(trees_gdf)}")

    # Apply mapping
    trees_gdf["lumped_species"] = trees_gdf["species_code"].map(species_mapping)
    trees_gdf["is_primary"] = trees_gdf["species_code"].map(primary_mapping)

    # Filter to only primary species at that lumped level
    trees_before = len(trees_gdf)
    trees_gdf = trees_gdf[trees_gdf["is_primary"] == True].copy()
    trees_after = len(trees_gdf)

    print(f"Trees removed (non-primary at {lump_level}): {trees_before - trees_after}")
    print(f"Trees kept (primary at {lump_level}): {trees_after}")
    print(f"Lumped species count : {trees_gdf['lumped_species'].nunique()}")

    # Show the mapping summary
    print(f"\nSpecies lumping summary:")
    lump_summary = trees_gdf.groupby("lumped_species")["species_code"].apply(
        lambda x: f"{len(x)} trees from {x.nunique()} original species"
    )
    for lumped, desc in lump_summary.items():
        print(f"  {lumped}: {desc}")

    return trees_gdf, crosswalk


def create_species_quantity_matrix(trees_gdf, plot_ids, plot_weights, use_lumped=True):
    """
    Create a matrix of tree counts per species per plot, weighted by number of dataset pairs.

    Args:
        trees_gdf: GeoDataFrame of individual trees
        plot_ids: List of plot IDs
        plot_weights: Dictionary mapping plot_id to weight (number of pairs)
        use_lumped: If True, use 'lumped_species' column, else use 'species_code'
    """
    # Filter trees to only include plots we're working with
    trees_filtered = trees_gdf[trees_gdf["plot_id"].isin(plot_ids)].copy()

    species_col = "lumped_species" if use_lumped else "species_code"

    print(f"Creating species matrix using: {species_col}")
    print(f"Found {len(trees_filtered)} individual trees across {len(plot_ids)} plots")

    # Count trees by plot and species
    species_counts = (
        trees_filtered.groupby(["plot_id", species_col])
        .size()
        .reset_index(name="tree_count")
    )

    # Pivot to create matrix: rows = plots, columns = species, values = tree counts
    species_matrix = species_counts.pivot(
        index="plot_id", columns=species_col, values="tree_count"
    ).fillna(0)

    # Reindex to match plot order and convert to integer
    species_matrix = species_matrix.reindex(plot_ids).fillna(0).astype(int)

    # Apply weights - multiply each plot's counts by its weight
    for plot_id in species_matrix.index:
        weight = plot_weights.get(plot_id, 1)
        species_matrix.loc[plot_id] = species_matrix.loc[plot_id] * weight

    print(
        f"Species quantity matrix: {len(species_matrix)} plots x {len(species_matrix.columns)} species"
    )
    print(f"Matrix weighted by number of drone pairs per plot")

    return species_matrix


def find_best_split_with_trees(
    plots_gdf, species_matrix, plot_weights, val_frac=VAL_FRACTION, seed=SEED
):
    """
    Find the best train/val split that minimizes species composition differences.
    Uses weighted contributions based on number of drone pairs per plot.
    """
    np.random.seed(seed)

    plot_ids = plots_gdf["plot_id"].tolist()

    # Calculate target number of datasets for validation (not plots)
    total_pairs = sum(plot_weights.values())
    target_val_pairs = int(total_pairs * val_frac)

    best_split = None
    best_score = float("inf")

    # Try a large number of random splits to find the best one
    print(f"\nFinding optimal split:")
    print(f"Total dataset pairs: {total_pairs}")
    print(f"Target validation pairs: {target_val_pairs} ({val_frac:.0%})")
    print(f"Testing 100000 different train/val combinations...")

    for i in range(100000):
        if i % 200 == 0 and i > 0:
            print(f"Tested {i} combinations (best score so far: {best_score:.4f})")

        # Randomly shuffle and select plots until we reach target validation pairs
        shuffled_plots = np.random.permutation(plot_ids)
        val_plots = []
        val_pairs_count = 0

        for plot_id in shuffled_plots:
            if val_pairs_count >= target_val_pairs:
                break
            val_plots.append(plot_id)
            val_pairs_count += plot_weights[plot_id]

        train_plots = [plot_id for plot_id in plot_ids if plot_id not in val_plots]

        # Calculate total species quantity for this split (already weighted)
        train_species_total = species_matrix.loc[train_plots].sum()
        val_species_total = species_matrix.loc[val_plots].sum()

        # Convert to proportions for fair comparison
        total_train = train_species_total.sum()
        total_val = val_species_total.sum()

        if total_train > 0 and total_val > 0:
            train_props = train_species_total / total_train
            val_props = val_species_total / total_val

            # Calculate difference score (lower is better)
            score = ((train_props - val_props) ** 2).sum()
        else:
            score = float("inf")

        if score < best_score:
            best_score = score
            best_split = (train_plots, val_plots)

    print(f"Best balance score: {best_score:.4f}")
    return best_split


def analyze_split_quality_with_trees(plots_gdf, species_matrix, plot_weights):
    """
    Analyze the quality of the train/val split using weighted contributions.
    """
    train_plots = plots_gdf[plots_gdf["split"] == "train"]["plot_id"].tolist()
    val_plots = plots_gdf[plots_gdf["split"] == "val"]["plot_id"].tolist()

    # Calculate number of pairs in each split
    train_pairs = sum(plot_weights[pid] for pid in train_plots)
    val_pairs = sum(plot_weights[pid] for pid in val_plots)
    total_pairs = train_pairs + val_pairs

    train_species_total = species_matrix.loc[train_plots].sum()
    val_species_total = species_matrix.loc[val_plots].sum()

    # Convert to proportions
    total_train_trees = train_species_total.sum()
    total_val_trees = val_species_total.sum()
    train_props = train_species_total / total_train_trees
    val_props = val_species_total / total_val_trees

    print(f"\nSPLIT QUALITY ANALYSIS (Using Weighted Individual Tree Data)")
    print(f"Train plots: {len(train_plots)} ({len(train_plots)/len(plots_gdf):.1%})")
    print(f"Val plots: {len(val_plots)} ({len(val_plots)/len(plots_gdf):.1%})")
    print(f"Train pairs: {train_pairs} ({train_pairs/total_pairs:.1%})")
    print(f"Val pairs: {val_pairs} ({val_pairs/total_pairs:.1%})")
    print(
        f"Train weighted trees: {total_train_trees:,.0f} ({total_train_trees/(total_train_trees + total_val_trees):.1%})"
    )
    print(
        f"Val weighted trees: {total_val_trees:,.0f} ({total_val_trees/(total_train_trees + total_val_trees):.1%})"
    )

    # Species composition comparison
    print(f"\nSpecies composition (proportion of total weighted trees in each split):")
    comparison = pd.DataFrame(
        {
            "Train_Weighted_Count": train_species_total,
            "Val_Weighted_Count": val_species_total,
            "Train_Prop": train_props,
            "Val_Prop": val_props,
            "Difference": val_props - train_props,
        }
    )

    comparison["Train_Prop"] = comparison["Train_Prop"].round(3)
    comparison["Val_Prop"] = comparison["Val_Prop"].round(3)
    comparison["Difference"] = comparison["Difference"].round(3)

    # Sort by total quantity
    comparison = comparison.loc[
        (comparison["Train_Weighted_Count"] + comparison["Val_Weighted_Count"])
        .sort_values(ascending=False)
        .index
    ]

    print(comparison)

    # Overall balance metrics
    balance_score = ((train_props - val_props) ** 2).sum()
    print(f"\nOverall species balance score (lower is better): {balance_score:.4f}")

    abs_diffs = comparison["Difference"].abs()
    if len(abs_diffs) > 0:
        print(
            f"Largest species proportion difference: {abs_diffs.max():.3f} ({abs_diffs.idxmax()})"
        )
        print(f"Average species proportion difference: {abs_diffs.mean():.3f}")

    return comparison


def visualize_split_with_trees(plots_gdf, species_matrix, plot_weights, lump_level):
    """
    Visualize the train/val split geographically and by species composition.
    """
    # Plot 1: Geographic split
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    train_gdf = plots_gdf[plots_gdf["split"] == "train"]
    val_gdf = plots_gdf[plots_gdf["split"] == "val"]

    train_pairs = sum(plot_weights[pid] for pid in train_gdf["plot_id"])
    val_pairs = sum(plot_weights[pid] for pid in val_gdf["plot_id"])

    train_gdf.plot(
        ax=ax1, color="green", alpha=0.7, edgecolor="darkgreen", linewidth=0.5
    )
    val_gdf.plot(ax=ax1, color="red", alpha=0.7, edgecolor="darkred", linewidth=0.5)

    ax1.set_title("Train-Val Split")
    train_patch = mpatches.Patch(
        color="green", label=f"Train ({len(train_gdf)} plots, {train_pairs} pairs)"
    )
    val_patch = mpatches.Patch(
        color="red", label=f"Val ({len(val_gdf)} plots, {val_pairs} pairs)"
    )
    ax1.legend(handles=[train_patch, val_patch])

    plt.tight_layout()
    plt.show()

    # Plot 2: Species composition comparison
    train_plots = train_gdf["plot_id"].tolist()
    val_plots = val_gdf["plot_id"].tolist()

    train_species_total = species_matrix.loc[train_plots].sum()
    val_species_total = species_matrix.loc[val_plots].sum()

    total_train_trees = train_species_total.sum()
    total_val_trees = val_species_total.sum()
    train_props = train_species_total / total_train_trees
    val_props = val_species_total / total_val_trees

    # Get all species, sorted by total quantity
    total_quantity = train_species_total + val_species_total
    all_species = total_quantity.sort_values(ascending=False).index

    # Adjust figure size based on number of species
    n_species = len(all_species)
    fig_width = max(
        12, n_species * 0.8
    )  # At least 12 inches, or scale with species count
    fig2, ax2 = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(n_species)
    width = 0.35

    bars_train = ax2.bar(
        x - width / 2,
        train_props[all_species],
        width,
        label="Train",
        color="green",
        alpha=0.7,
    )
    bars_val = ax2.bar(
        x + width / 2,
        val_props[all_species],
        width,
        label="Val",
        color="red",
        alpha=0.7,
    )

    # Label each bar with weighted counts
    for bar, species in zip(bars_train, all_species):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{int(train_species_total[species]):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="green",
            fontweight="bold",
            rotation=90,
        )

    for bar, species in zip(bars_val, all_species):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{int(val_species_total[species]):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
            fontweight="bold",
            rotation=90,
        )

    ax2.set_xlabel("Species (Lumped)", fontsize=12)
    ax2.set_ylabel("Proportion of Total Weighted Trees", fontsize=12)
    ax2.set_title(
        f"Species Composition: Train vs Val ({lump_level} lumping)", fontsize=14
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_species, rotation=90, ha="center", fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # TODO: There is another step to be added before this where we filter out the withheld test-plots datasets
    # from the list that will be containing train & val datasets. The below file is a result of that.
    pairs_df = pd.read_csv(
        "/ofo-share/scratch/amritha/tree-species-scratch/november-run/train_and_val_dsets.csv"
    )
    pairs_df["plot_id"] = pairs_df["plot_id"].apply(lambda x: f"{int(x):04d}")

    # Calculate weights: number of drone pairs assigned per plot
    plot_weights = pairs_df["plot_id"].value_counts().to_dict()

    print(f"Plot pair counts:")
    for plot_id, count in sorted(plot_weights.items()):
        if count > 1:
            print(f"  Plot {plot_id}: {count} pairs")
    print(f"Total number of unique plots: {len(plot_weights)}")
    print(f"Total number of datasets: {sum(plot_weights.values())}")
    print(
        f"Plots with multiple drone pairs: {sum(1 for c in plot_weights.values() if c > 1)}"
    )

    plots_gdf = gpd.read_file(path_config.ground_reference_plots_file)[
        ["plot_id", "geometry"]
    ]
    plots_gdf = plots_gdf[plots_gdf["plot_id"].isin(pairs_df["plot_id"])].copy()

    trees_gdf = gpd.read_file(path_config.ground_reference_trees_file)
    trees_gdf["plot_id"] = trees_gdf["plot_id"].apply(lambda x: f"{int(x):04d}")

    # Apply species lumping using crosswalk
    trees_gdf, crosswalk = load_and_apply_crosswalk(
        trees_gdf, CROSSWALK_PATH, LUMP_LEVEL
    )

    # After filtering to primary species, some plots might have no trees left
    # Filter plots to only those that still have trees
    plots_with_trees = trees_gdf["plot_id"].unique()
    plots_gdf = plots_gdf[plots_gdf["plot_id"].isin(plots_with_trees)].copy()

    # Update plot_weights to only include plots with remaining trees
    plot_weights = {
        pid: weight for pid, weight in plot_weights.items() if pid in plots_with_trees
    }

    print(f"After filtering to {LUMP_LEVEL} primary species:")
    print(f"Plots remaining: {len(plots_gdf)}")
    print(f"Total datasets remaining: {sum(plot_weights.values())}")

    # Create species quantity matrix with lumped species
    species_matrix = create_species_quantity_matrix(
        trees_gdf, plots_gdf["plot_id"].tolist(), plot_weights, use_lumped=True
    )
    species_matrix.to_csv(f"species_distribution_matrix_weighted_{LUMP_LEVEL}.csv")
    total_quantity = species_matrix.sum()
    print(f"\nTotal weighted trees per lumped species:")
    print(total_quantity.sort_values(ascending=False))

    # Find best split using lumped species
    print(
        f"\nFinding optimal train/val split (target: {VAL_FRACTION:.0%} of pairs in val)..."
    )
    train_plot_ids, val_plot_ids = find_best_split_with_trees(
        plots_gdf, species_matrix, plot_weights
    )

    # Assign splits
    plots_gdf["split"] = "train"
    plots_gdf.loc[plots_gdf["plot_id"].isin(val_plot_ids), "split"] = "val"

    # Analyze results
    comparison = analyze_split_quality_with_trees(
        plots_gdf, species_matrix, plot_weights
    )

    # Save results - ALL pairs for a plot get the same split assignment
    pairs_df = pairs_df.merge(plots_gdf[["plot_id", "split"]], on="plot_id", how="left")

    # Check for any datasets that didn't get a split assigned (plots with no primary species)
    missing_splits = pairs_df["split"].isna().sum()
    if missing_splits > 0:
        print(
            f"\nWARNING: {missing_splits} dataset(s) has no split assignment (plots with no primary species at {LUMP_LEVEL})"
        )
        print(pairs_df[pairs_df["split"].isna()])

    pairs_df.to_csv(path_config.train_val_split_file, index=False)
    plots_gdf.to_file(path_config.train_val_split_gpkg_file)

    # Verify all datasets for same plot have same split assigned
    print("\nVerifying split consistency...")
    split_check = pairs_df.groupby("plot_id")["split"].nunique()
    if (split_check <= 1).all():
        print("All pairs for each plot are in the same split")
    else:
        print("WARNING: Some plots have pairs in different splits!")
        print(split_check[split_check > 1])

    visualize_split_with_trees(plots_gdf, species_matrix, plot_weights, LUMP_LEVEL)

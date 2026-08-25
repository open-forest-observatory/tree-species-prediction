# Purpose: Using the ground-drone pairings that pass all quality filters (e.g, aligned, correct
# altitudes, correct image subsets), select a subset of them (with ground plots as consistent as
# possible in terms of survey protocl and species composition) for prototypting.

library(tidyverse)
library(sf)


DATASETS_QUALIFIED_FILEPATH = "/ofo-share/species-prediction-project/intermediate/dataset_to_render.csv"
PLOT_TREE_DATA = "/ofo-share/species-prediction-project/raw/ground-reference/ofo_ground-reference_trees.gpkg"

MISSIONS_UNSYNCED_IMAGES = c( "000002", "000005", "000008", "000010", "000011", "000013", 
    "000015", "000016", "000018", "000019", "000020", "000021", "000022", 
    "000023", "000024", "000028", "000029", "000030", "000031", "000032", 
    "000033", "000034", "000035", "000036", "000040", "000041", "000044", 
    "000045", "000046", "000047", "000048", "000052", "000055", "000056", 
    "000058", "000059", "000060", "000061", "000062", "000065", "000067", 
    "000069", "000070", "000071", "000072", "000073", "000074", "000075", 
    "000076", "000078", "000079", "000081", "000082", "000083", "000084", 
    "000085", "000086", "000087", "000089", "000090", "000091", "000092", 
    "000093", "000096", "000097", "000100", "000101", "000103", "000104", 
    "000106", "000110", "000113", "000118", "000119", "000121", "000124", 
    "000127", "000128", "000129", "000131", "000133", "000139", "000145", 
    "000147", "000148", "000149", "000151", "000152", "000153", "000155", 
    "000157", "000158", "000160", "000162", "000164", "000165", "000168", 
    "000169", "000173", "000174", "000175", "000177", "000179", "000182", 
    "000203", "000205", "000208", "000210", "000214", "000227", "000231", 
    "000234", "000236", "000238", "000239", "000240", "000242", "000244", 
    "000245", "000246", "000248", "000251", "000254", "000256", "000257", 
    "000259", "000261", "000262", "000263", "000266", "000267", "000268", 
    "000271", "000272", "000273", "000275", "000277", "000323", "000324", 
    "000333", "000334", "000337", "000339", "000340", "000341", "000342", 
    "000346", "000348", "000349", "000352", "000353", "000370", "000371", 
    "000372", "000373", "000374", "000375", "000376", "000377", "000378", 
    "000379", "000380", "000382", "000384", "000385", "000386", "000388", 
    "000390", "000391", "000392", "000393", "000394", "000395", "000396", 
    "000398", "000400", "000401", "000402", "000403", "000404", "000405", 
    "000406", "000407", "000408", "000409", "000410", "000411", "000413", 
    "000414", "000415", "000416", "000417", "000419", "000420", "000426", 
    "000435", "000436", "000440", "000442", "000444", "000446", "000448", 
    "000450", "000451", "000452", "000479", "000544", "000549", "000551", 
    "000559", "000563", "000611")

PLOTS_CONSISTENT_PROTOCOL = c(
    # VP and TNC:
    1:51, 88:111,
    # STEF:
    73:80,
    # Blodgett:
    115:118
) |> str_pad(width = 4, side = "left", pad = "0")


# Get qualified, aligned datasets
datasets_qualified = read_csv(DATASETS_QUALIFIED_FILEPATH) |>
  filter(good_shift == 1) |>
  pull(dataset)

# Of the plots with consistent protocol, summarize species comp, tree density, tree size, and select
# the most consistent plots

trees = st_read(PLOT_TREE_DATA) |>
    filter(plot_id %in% PLOTS_CONSISTENT_PROTOCOL)

# Look for outliers
hist(trees$dbh)
trees$dbh |>
    unique() |>
    sort()

# Tree diameter typos? Exclude those trees
trees = trees |>
    filter(dbh < 250)

# Only use live trees for purposes of plot filtering
trees = trees |>
    filter(live_dead == "L") #dbh > 25, 

# Get stand structure and species comp summaries to narrow on most consistent plots
trees_summ = trees |>
    group_by(plot_id) |>
    mutate(n_trees = n()) |>
    summarize(
        n_trees = n(),
        pct_PIPJ = sum(species_code %in% c("PIPO", "PIPJ", "PIJE"), na.rm=TRUE) / n(),
        pct_ABCO = sum(species_code %in% c("ABCO"), na.rm=TRUE) / n(),
        pct_PSME = sum(species_code %in% c("PSME"), na.rm=TRUE) / n(),
        pct_PILA = sum(species_code %in% c("PILA"), na.rm=TRUE) / n(),
        pct_NODE3 = sum(species_code %in% c("NODE3"), na.rm=TRUE) / n(),
        pct_PICO = sum(species_code %in% c("PICO"), na.rm=TRUE) / n(),
        pct_CADE27 = sum(species_code %in% c("CADE27"), na.rm=TRUE) / n(),
        pct_othersp = sum(!species_code %in% c("PIPO", "PIPJ", "PIJE", "ABCO", "PSME", "PILA", "NODE3", "PICO", "CADE27"), na.rm=TRUE) / n(),
        pct_mainsp = sum(species_code %in% c("PIPO", "PIPJ", "PIJE", "ABCO", "PSME", "PILA", "CADE27"), na.rm=TRUE) / n(),
        qmd = sqrt(mean(dbh^2, na.rm=TRUE))
    )

hist(trees_summ$pct_mainsp)
hist(trees_summ$n_trees)
hist(trees_summ$qmd)

# Select plots with average tree conditions
trees_summ_filt = trees_summ |>
    filter(
        pct_mainsp > 0.8,
        pct_NODE3 == 0.0,
        between(qmd, 50, 70)
    )

# IDs of the plots with consistent protocol and average tree conditions
central_plots = trees_summ_filt$plot_id


# Filter the qualifying datasets to only include those with consistent protocol, average tree
# conditions, and no unsynced images

has_unsynced_images = str_detect(datasets_qualified, paste(MISSIONS_UNSYNCED_IMAGES, collapse = "|"))
is_central_plot = str_sub(datasets_qualified, start = 1, end = 4) %in% central_plots

datasets_central = datasets_qualified[!has_unsynced_images & is_central_plot]

# Sort in reverse order so we can remove the first plot of any duplicates
datasets_central = datasets_central[order(datasets_central, decreasing = TRUE)]

# Get the plot ID out of the dataset ID and find duplicates
plots_central = str_extract(datasets_central, "[:digit:]{4}")
is_duplicated = duplicated(plots_central)

# Remove the datasets that have duplicated plots, keeping the most recent one, and sort back
# into ascending order
datasets_final = datasets_central[!is_duplicated] |>
    sort()

# Exclude the two that are not fro the STEF project
datasets_exclude = c("0091_001415_001414", "0118_000649_000650")
datasets_final = datasets_final[!datasets_final %in% datasets_exclude]

# Make a data frame with one row for each plot in plots_final, with rows pulled from tree_summ_filt,
# with the plots that occur multiple times in data_set final occurring the same number of times in
# the final data frame
final_datasets_attributes = tibble(dataset = datasets_final) |>
    mutate(plot_id = str_extract(dataset, "[:digit:]{4}")) |>
    left_join(trees_summ_filt, by = "plot_id") |>
    arrange(plot_id)

tot_trees = sum(final_datasets_attributes$n_trees)
tot_trees

datasets_final

# Save the final selected datasets
write_csv(
    tibble(dataset = datasets_final),
    "/ofo-share/species-prediction-project/intermediate/dataset_to_render_subset.csv"
)

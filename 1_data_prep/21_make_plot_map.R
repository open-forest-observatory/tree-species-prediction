# Purpose: Make a map figure of the study area with plot locations. Obtain the list of included
# plots (in train, val, test) from the metadata files accompanying the image chips.

library(rnaturalearth)
library(ggspatial)
library(patchwork)
library(ceramic)
library(tidyterra)
library(tidyverse)
library(sf)

TRAIN_IMAGE_METADATA_FILEPATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/train_metadata.csv"
TEST_IMAGE_METADATA_FILEPATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/test_metadata.csv"
VAL_IMAGE_METADATA_FILEPATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/val_metadata.csv"

YUBA_AREA_FILEPATH = "/ofo-share/project-data/species-prediction-project/raw/north_yuba_area.kml"

PLOTS_METADATA_FILEPATH = "/ofo-share/project-data/species-prediction-project/raw/ground-reference/ofo_ground-reference_plots.gpkg"

MAP_FIGURE_FILEPATH = "/ofo-share/project-data/species-prediction-project/figures/plots-centroids-map.png"

# Inset map extent (regional context), degrees WGS84 -- tweak these to adjust the inset view
INSET_XMIN = -121.5
INSET_XMAX = -119.5
INSET_YMIN = 38.5
INSET_YMAX = 40

test_metadata = read_csv(TEST_IMAGE_METADATA_FILEPATH)
val_metadata = read_csv(VAL_IMAGE_METADATA_FILEPATH)


test_datasets = unique(test_metadata$dataset_name)
val_datasets = unique(val_metadata$dataset_name)

test_datasets = data.frame(type = "test", name = test_datasets)
val_datasets = data.frame(type = "val", name = val_datasets)

datasets = rbind(test_datasets, val_datasets)

datasets = datasets |>
  mutate(plot_id = str_split(name, "_") |>
           map_chr(1))


plots_metadata = st_read(PLOTS_METADATA_FILEPATH)

# Bind type onto the plots
plots_metadata = plots_metadata |>
  left_join(datasets, by = c("plot_id" = "plot_id"))

# Get centroids
plots_centroids = st_centroid(plots_metadata)



# Get state/province boundaries (US states + Canadian provinces)
us_states = ne_states(country = "united states of america", returnclass = c("sf"))
ca_provinces = ne_states(country = "canada", returnclass = c("sf"))
states = bind_rows(us_states, ca_provinces)

# Get Yuba area polygon
yuba_area = st_read(YUBA_AREA_FILEPATH) |> st_transform(4326)

# Transform plot centroids to WGS84 for plotting
plots_centroids_wgs84 = st_transform(plots_centroids, 4326)

# Get bbox of plot centroids with buffer
footprints_bbox = st_bbox(plots_centroids_wgs84)
bbox_buffer = 1 # degrees (~100 km)

# Create a bbox polygon for the main map extent to show in the inset
main_extent_bbox = st_bbox(c(
  xmin = as.numeric(footprints_bbox["xmin"]) - bbox_buffer,
  ymin = as.numeric(footprints_bbox["ymin"]) - bbox_buffer,
  xmax = as.numeric(footprints_bbox["xmax"]) + bbox_buffer,
  ymax = as.numeric(footprints_bbox["ymax"]) + bbox_buffer
), crs = st_crs(4326)) |> 
  st_as_sfc()

# Create a bbox polygon for the inset map extent
inset_extent_bbox = st_bbox(c(
  xmin = INSET_XMIN,
  ymin = INSET_YMIN,
  xmax = INSET_XMAX,
  ymax = INSET_YMAX
), crs = st_crs(4326)) |>
  st_as_sfc()

# Prepare data for basemaps (transform to Web Mercator EPSG:3857)
plots_centroids_3857 = st_transform(plots_centroids_wgs84, 3857)
main_extent_3857 = st_transform(main_extent_bbox, 3857)
inset_extent_3857 = st_transform(inset_extent_bbox, 3857)

# Get basemap for main map
basemap_ca = ceramic::cc_location(loc = main_extent_3857, type = "mapbox.satellite")

# Get basemap for inset map
basemap_inset = ceramic::cc_location(loc = inset_extent_3857, type = "mapbox.satellite")

# Get basemap for main map
main_map_extent = st_as_sf(main_extent_bbox)
main_map_extent_3857 = st_transform(main_map_extent, 3857)
basemap_main = ceramic::cc_elevation(loc = main_map_extent_3857)

# Transform states for plotting
states_3857 = st_transform(states, 3857)

# Inset map: regional context
ca_inset = ggplot() +
  geom_spatraster_rgb(data = basemap_inset, alpha = 0.5, maxcell = Inf) +
  geom_sf(data = yuba_area, fill = NA, color = "red", linewidth = 0.8) +
  geom_sf(data = states_3857, fill = NA, linewidth = 0.4, color = "black") +
  geom_sf(data = plots_centroids_3857, color = "white", size = 2) +
  geom_sf(data = plots_centroids_3857, color = "#E8A735", size = 1.5) +
  coord_sf(crs = 4326, expand = FALSE,
           xlim = c(INSET_XMIN, INSET_XMAX),
           ylim = c(INSET_YMIN, INSET_YMAX)) +
  theme_bw(12) +
  theme(panel.grid = element_blank(),
        axis.title = element_blank(),
        plot.title = element_text(margin = margin(t = 0))) +
  labs(title = "(b)")

# Main map of plot centroids with basemap
plots_centroids_map = ggplot() +
  geom_spatraster_rgb(data = basemap_ca, alpha = 0.5, maxcell = Inf) +
  scale_fill_viridis_c(name = "Elev. (m)", breaks = seq(0, 3500, 500)) +
  geom_sf(data = inset_extent_3857, fill = NA, color = "blue", linewidth = 0.8) +
  geom_sf(data = states_3857, fill = NA, linewidth = 0.4, color = "black") +
  geom_sf(data = plots_centroids_3857, color = "white", size = 2) +
  geom_sf(data = plots_centroids_3857, color = "#E8A735", size = 1.5) +
  coord_sf(crs = 4326, expand = FALSE,
           xlim = c(footprints_bbox["xmin"] - bbox_buffer, footprints_bbox["xmax"] + bbox_buffer),
           ylim = c(footprints_bbox["ymin"] - bbox_buffer, footprints_bbox["ymax"] + bbox_buffer)) +
#   scale_y_continuous(breaks = seq(39.35, 39.85, 0.1)) +
  scale_x_continuous(breaks = scales::breaks_width(4)) +
  theme_bw(12) +
#   annotation_scale(pad_x = unit(0.6, "cm"),
#                    pad_y = unit(0.6, "cm"), 
#                    location = "tr", 
#                    text_cex = 1, 
#                    bar_cols = c("black", "black"),
#                    height = unit(0.01, "cm")) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(margin = margin(t = 0))) +
  labs(title = "(a)")

combined_map = plots_centroids_map + ca_inset +
  plot_layout(widths = c(1, 2))

png(MAP_FIGURE_FILEPATH,
    res = 500, width = 3000, height = 1800)
print(combined_map)
dev.off()


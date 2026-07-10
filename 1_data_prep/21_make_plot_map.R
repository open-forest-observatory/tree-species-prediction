# Purpose: Make a map figure of the study area with plot locations. Obtain the list of included
# plots (in train, val, test) from the metadata files accompanying the image chips.

library(rnaturalearth)
library(ggspatial)
library(patchwork)
library(ceramic)
library(tidyterra)
library(tidyverse)
library(sf)

TRAIN_VAL_SPLIT_FILEPATH = "/ofo-share/project-data/species-prediction-project/intermediate/train_val_split_l2.csv"
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


train_val_split = read_csv(TRAIN_VAL_SPLIT_FILEPATH)
train_plots = unique(train_val_split$plot_id[train_val_split$split == "train"])


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

train_datasets = data.frame(type = "train", plot_id = train_plots)

datasets = bind_rows(datasets, train_datasets)


plots_metadata = st_read(PLOTS_METADATA_FILEPATH)

# Bind type onto the plots
plots_metadata = plots_metadata |>
  left_join(datasets, by = c("plot_id" = "plot_id"))

# Drop plots that are not in train, val, or test datasets
plots_metadata = plots_metadata |>
  filter(!is.na(type))


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
plots_centroids_3857 = st_transform(plots_centroids_wgs84, 3857) |>
  mutate(type = factor(type, levels = c("train", "val", "test"),
                       labels = c("Train", "Validation", "Test")))
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

# Shared color scale for both panels: plot types (3 stops along the viridis scale) plus the two
# extent polygons, so a single legend covers everything
type_colors = viridisLite::viridis(3, begin = 0.0, end = 0.8)
# Navy: between the pure blue formerly used for the bbox in (a) and the near-black panel border
# of (b); used for both so they read as the same area
inset_color = "#0000B3"
yuba_color = "#CC0000"
legend_colors = c(
  "Train" = type_colors[1],
  "Validation" = type_colors[2],
  "Test" = type_colors[3],
  "Detail area" = inset_color,
  "North Yuba area" = yuba_color
)

shared_color_scale = scale_color_manual(
  name = NULL,
  values = legend_colors,
  limits = names(legend_colors),
  guide = guide_legend(
    override.aes = list(
      shape = c(16, 16, 16, NA, NA),
      size = c(2, 2, 2, NA, NA),
      linetype = c("blank", "blank", "blank", "solid", "solid"),
      linewidth = c(NA, NA, NA, 0.8, 0.8),
      fill = NA,
      alpha = 1
    )
  )
)

# Inset map: regional context
ca_inset = ggplot() +
  geom_spatraster_rgb(data = basemap_inset, alpha = 0.5, maxcell = Inf) +
  geom_sf(data = yuba_area, aes(color = "North Yuba area"), fill = NA, linewidth = 0.8) +
  geom_sf(data = states_3857, fill = NA, linewidth = 0.4, color = "black") +
  # Invisible layer (alpha = 0; restored to visible in the legend by override.aes) so this
  # panel's legend gets a key for the inset-area box that is drawn in panel (a)
  geom_sf(data = inset_extent_3857, aes(color = "Detail area"), fill = NA,
          linewidth = 0.8, alpha = 0) +
  geom_sf(data = plots_centroids_3857, color = "white", size = 2) +
  geom_sf(data = plots_centroids_3857, aes(color = type), size = 1.5) +
  shared_color_scale +
  coord_sf(crs = 4326, expand = FALSE,
           xlim = c(INSET_XMIN, INSET_XMAX),
           ylim = c(INSET_YMIN, INSET_YMAX)) +
  theme_bw(11) +
  theme(panel.grid = element_blank(),
        axis.title = element_blank(),
        # Match the panel outline to the bbox showing this panel's extent in panel (a); 2x the
        # bbox linewidth because the half of the border outside the panel edge is clipped
        panel.border = element_rect(color = inset_color, fill = NA, linewidth = 1.6),
        plot.title = element_text(margin = margin(t = 0))) +
  labs(title = "(b)")

# Main map of plot centroids with basemap
plots_centroids_map = ggplot() +
  geom_spatraster_rgb(data = basemap_ca, alpha = 0.5, maxcell = Inf) +
  scale_fill_viridis_c(name = "Elev. (m)", breaks = seq(0, 3500, 500)) +
  geom_sf(data = inset_extent_3857, aes(color = "Detail area"), fill = NA, linewidth = 0.8) +
  geom_sf(data = states_3857, fill = NA, linewidth = 0.4, color = "black") +
  geom_sf(data = plots_centroids_3857, color = "white", size = 2) +
  geom_sf(data = plots_centroids_3857, aes(color = type), size = 1.5) +
  shared_color_scale +
  coord_sf(crs = 4326, expand = FALSE,
           xlim = c(footprints_bbox["xmin"] - bbox_buffer, footprints_bbox["xmax"] + bbox_buffer),
           ylim = c(footprints_bbox["ymin"] - bbox_buffer, footprints_bbox["ymax"] + bbox_buffer)) +
#   scale_y_continuous(breaks = seq(39.35, 39.85, 0.1)) +
  scale_x_continuous(breaks = scales::breaks_width(4)) +
  theme_bw(11) +
#   annotation_scale(pad_x = unit(0.6, "cm"),
#                    pad_y = unit(0.6, "cm"), 
#                    location = "tr", 
#                    text_cex = 1, 
#                    bar_cols = c("black", "black"),
#                    height = unit(0.01, "cm")) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(margin = margin(t = 0)),
        # Suppress this panel's legend; the shared legend is built in panel (b) and
        # collected by patchwork for the whole figure
        legend.position = "none") +
  labs(title = "(a)")

combined_map = plots_centroids_map + ca_inset +
  plot_layout(widths = c(1, 2), guides = "collect")

png(MAP_FIGURE_FILEPATH,
    res = 500, width = 3000, height = 1800)
print(combined_map)
dev.off()


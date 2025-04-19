'''

COSC527 - RxFire Project - Data Generation Script

Combines the raw datafiles into a stacked GeoTIFF

'''
import raster_functions as rf
import os


DATA_DIR = "data/raw/"
if os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

# Bounding boxes
tn_bbox = [-84,35.5,-83.8,35.7]
ca_bbox = [-118.7,34.05,-118.6,34.15]

# Crop data
rf.crop_raster("tn_elevation.tif", "cropped_tn_elevation.tif", tn_bbox)
rf.crop_raster("ca_elevation.tif", "cropped_ca_elevation.tif", ca_bbox)
rf.crop_raster("NDVI_Tennessee.tif", "cropped_tn_ndvi.tif", tn_bbox)
rf.crop_raster("NDVI_California.tif", "cropped_ca_ndvi.tif", ca_bbox)

# Compute terrain parameters
rf.compute_parameters("cropped_tn_elevation.tif", "tn")
rf.compute_parameters("cropped_ca_elevation.tif", "ca")

# Stack data
rf.stack_rasters(
    ["cropped_tn_ndvi.tif", "cropped_tn_elevation.tif", "tn_slope.tif", "tn_aspect.tif"], 
    "../stacked_tn.tif", 
    metadata={"AMBTEMP": 7.778, "RELHUM": 0.6, "WIND": "-0.894,0", "SMAP": 0.23, "BBOX": ",".join(map(str, tn_bbox))}
)
print("Saved TN Dataset to data/stacked_tn.tif")

rf.stack_rasters(
    ["cropped_ca_ndvi.tif", "cropped_ca_elevation.tif", "ca_slope.tif", "ca_aspect.tif"], 
    "../stacked_ca.tif",
    metadata={"AMBTEMP": 21.111, "RELHUM": 0.7, "WIND": "0,4.783", "SMAP": 0.09, "BBOX": ",".join(map(str, ca_bbox))}
)
print("Saved CA Dataset to data/stacked_ca.tif")
###############
### IMPORTS ###
###############

from osgeo import gdal
import matplotlib.pyplot as plt

gdal.UseExceptions()

#################
### FUNCTIONS ###
#################

def crop_raster(input_file, output_file, bbox):
    """
    Crops a GeoTIFF file with metadata containing geospatial info.

    Parameters
    ----------
    input_file : str
        Path to input GeoTIFF to crop.
    output_file : str
        Path to output file to store cropped file.
    bbox : List[float]
        List containing latitude and longitude values to crop file to.
        Format is [min_lon, min_lat, max_lon, max_lat]
    """
    
    # GDAL does cropping for us. Use EPSG:4326 projection to keep data in latitude/longitude coordinates
    gdal.Warp(output_file, input_file, outputBounds=bbox, dstSRS='EPSG:4326')


def compute_parameters(input_file, output_file_prefix):
    """
    Computes slope and aspect from elevation data in GeoTIFF format.

    Parameters
    ----------
    input_file : str
        Path to input GeoTIFF with elevation data.
    output_file_prefix : str
        Prefix to add to output files.
    """

    # Compute aspect
    dem_options = gdal.DEMProcessingOptions(zeroForFlat=False, format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    gdal.DEMProcessing(f"{output_file_prefix}_aspect.tif", input_file, processing="aspect", options=dem_options)

    # Compute slope
    dem_options = gdal.DEMProcessingOptions(format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    gdal.DEMProcessing(f"{output_file_prefix}_slope.tif", input_file, processing="slope", options=dem_options)


def stack_rasters(input_files, output_file, metadata=None):
    """
    Stacks multiple single-band GeoTIFFs into one multi-band GeoTIFF.

    Parameters
    ----------
    input_files : List[str]
        List of paths to files to stack together.
    output_file : str
        Path to output file to store stacked data.
    """

    # Build a VRT (Virtual Raster) from input list, stacking as separate bands
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT('/vsimem/stacked.vrt', input_files, options=vrt_options)

    # if metadata is not None:
    #     vrt.SetMetadata(metadata)

    # Translate the VRT to a physical multi-band GeoTIFF
    gdal.Translate(output_file, vrt)

    # Clean up
    vrt = None  # Close VRT
    gdal.Unlink('/vsimem/stacked.vrt')  # Remove from memory


def extract_array(input_file, band_number=1):
    """
    Returns array form of a band of GeoTIFF data.

    Parameters
    ----------
    input_file : str
        Path to file to get data from.
    band_number : int, optional
        Band to get raster data from.
    """

    # Open the GeoTIFF file
    dataset = gdal.Open(input_file)
    
    # Extract band data as an array
    band = dataset.GetRasterBand(band_number)
    return band.ReadAsArray()


def visualize_data(input_file, band_number=1, cmap='terrain', vmn=0, vmx=100):
    """
    Statically visualize data in GeoTIFF.

    Parameters
    ----------
    input_file : str
        Path to GeoTIFF file to visualize.
    band_number : int
        Band of GeoTIFF data to visualize (default is 1).
    cmap : str, optional
        CMAP to use with matplotlib coloring (default is 'terrain').
    vmn : int, optional
        Sets minimum value limit for color gradient (default is 0).
    vmx : int, optional
        Sets maximum value limit for color gradient (default is 100).
    """

    # Open the GeoTIFF file
    dataset = gdal.Open(input_file)
    
    # Read the desired band
    band = dataset.GetRasterBand(band_number)
    array = band.ReadAsArray()

    plt.figure(figsize=(10, 6))
    plt.imshow(array, cmap=cmap, vmin=vmn, vmax=vmx)
    plt.show()


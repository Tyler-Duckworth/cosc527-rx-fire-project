# Simulating Prescribed Fire With Cellular Automata

This project simulates a prescribed fire and how it affects the environment when a forest fire occurs using cellular automata. It was part of an assignment for the COSC 527 course&ndash;*"Biologically Inspired Computing"*&ndash;at the University of Tennessee, Knoxville 

## TODO
- Add logging support (control with verbosity flag)
- 

## Setup
- **Python Version Used:** v3.13.3
This project was made with Anaconda. The virtual environment I used has been exported into `rxfire_environment.yml`. 

You can restore it with `conda env create -f rxfire_environment.yml`.

### GDAL
To interact with the raster data created for this project, we made use of [GDAL](https://gdal.org/en/stable/). I have a Windows machine, and installing GDAL on it was very difficult at the time and caused a lot of frustration. What I ended up using was this [geospatial-wheels](https://github.com/cgohlke/geospatial-wheels/) repository of pre-built Python Wheels by [Christoph Gohlke](https://github.com/cgohlke) and installing the `gdal-[...]-cp313-win_amd64.whl`. 

Inside the Conda environment, I ran the following:

```
> pip install /path/to/gdal-[...].whl
```

## Creating The Dataset
You can create the dataset used for this project by running the `src/data_generation.py` script. This will take the raw layers stored in `data/raw/` and combine them into the two GeoTIFF files for the two areas analyzed in this project. The resulting files (`stacked_ca.tif` and `stacked_tn.tif`) will be stored in `data/`.

You can run the file with the following command:

```
> python src/data_generation.py
```

## Running The Simulation
The `src/fire_simulation.py` script contains the main code for the simulation. You can run it as follows:

```
usage: fire_simulation [-h] [--hide-animation] [-d OUTPUT_DIR] [-o OUTPUT_FILE] [-g GENERATIONS] [-f SWITCH_FRAME] [data_file]

Simulates a prescribed fire using a generated GeoTIFF file. See file for more info.

positional arguments:
  data_file             Path to .tif file to run the simulation on

options:
  -h, --help            show this help message and exit
  --hide-animation      Hides the animation
  -d, --output-dir OUTPUT_DIR
                        Output directory to save intermediate data (defaults to "results")
  -o, --output-file OUTPUT_FILE
                        Name of file to save the animation to as an MP4 (defaults to "[tn|ca]_anim.mp4")
  -g, --generations GENERATIONS
                        Total number of generations to run the simulation over.
  -f, --switch-frame SWITCH_FRAME
                        Specifies which frame to switch from prescribed fires (Phase 1) to the forest fire (Phase 2)
```
### Debugging The Simulation
I've included a debug configuration for Visual Studio Code (`.vscode/launch.json`) that you can use to specifically run the simulation in debug mode. To run it:
1. Open the repository in Visual Studio Code.
2. Select the correct Python interpreter (should be the Conda environment, if you are using that). 
3. Click the Debug icon (bug and play button), and click the Green arrow next to "Debug Simulation"

## Experimental Setup
TBD


## Credits
- **Team Members:**
    - Tyler Duckworth
    - Gabriel LaBoy
    - Andy Zheng
- **Teacher:** Dr. Catherine Schumann
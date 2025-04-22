import numpy as np
from osgeo import gdal
from perlin_numpy import generate_perlin_noise_3d
from datetime import datetime
import os
import json 

MAX_BUILDINGS = 10
BUILDING_POOL_DIST = [0.5, 0.25, 0.2, 0.05]
BUILDING_POOL = [
    {"width": 5, "height": 7, "name": "Research Station", "limit": 20},
    {"width": 2, "height": 2, "name": "Fire Tower", "limit": 5},
    {"width": 2, "height": 3, "name": "Bathroom", "limit": 10},
    {"width": 30, "height": 30, "name": "Town", "limit": 1}
]
WIND_VARIABILITY = 2 # wind will be randomized with a magnitude of (-x, x)
DEFAULT_AMBTEMP = 68
DEFAULT_RELHUM = 0.5
DEFAULT_WIND = "0,0"
DEFAULT_SMAP = 0.05

class Environment:
    ''' 
    Represents the simulation environment 

    Available Raster Bands:
    - 1 - NDVI (Greenness Index, [-1, 1])
    - 2 - Elevation (??, [0, 2000))
    - 3 - Slope (degrees, [0, 90])
    - 4 - Aspect (degrees, [0, 360])
    '''
    def __init__(self, source_file: str, output_dir: str, seed: int, generations: int):
        self.seed = seed
        raw_file = gdal.Open(source_file)
        self.output_dir = output_dir
        output_file_name = source_file.replace(".tif", f'results_{datetime.now().strftime("%Y_%m_%d__%H%M%S")}.tif')
        self.output_file = f"{output_dir}/{output_file_name}"
        self.grid_size = (raw_file.RasterXSize, raw_file.RasterYSize)
        # Read in area metadata encoded into TIFF
        metadata = raw_file.GetMetadata()
        self.ambtemp = metadata.get("AMBTEMP", DEFAULT_AMBTEMP)
        self.relhum = metadata.get("RELHUM", DEFAULT_RELHUM)
        self.soil_moisture = metadata.get("SMAP", DEFAULT_SMAP)
        wind_arr = metadata.get("WIND", DEFAULT_WIND).split(",")
        self.wind = (float(wind_arr[0]), float(wind_arr[1])) 

        # background layer for plot
        self.bg = raw_file.GetRasterBand(1).ReadAsArray()
        self.elevation = raw_file.GetRasterBand(3).ReadAsArray()
        self.elevation[self.elevation < 0] = 0
        
        '''
        State grid - represents all the different cell's states
        Grid states:
        - -1 - Burned (fuel eaten)
        -  0 - Unburned 
        -  1 - Burning (reproducing)
        -  2 - Dying
        '''
        self.grid = np.zeros(self.grid_size, dtype=int)
        
        ''' Lifespan Grid - Stores the number of days a cell has lived '''
        self.lifespan_grid = np.zeros(self.grid_size, dtype=int)
        
        ''' Slope Grid - Stores slope as percentage (TODO: Figure out aspect)'''
        self.slope_grid = raw_file.GetRasterBand(3).ReadAsArray() / 100
        
        self.init_fuel_grid(self.bg)
        self.init_wind_grid((raw_file.RasterXSize, raw_file.RasterYSize, 2), generations)
        self.buildings = []

    def save(self, step=0):
        ''' Saves the current info (grid) to a file'''
        if not os.path.exists(f"{self.output_dir}/frames"):
            os.makedirs(f"{self.output_dir}/frames")
        np.save(f'{self.output_dir}/frames/frame_{step}.npy', self.grid)

    def load_buildings_from_file(self, file_path):
        with open(file_path, "r") as fp:
            buf = json.load(fp)
        for bldg in buf["buildings"]:
            x, y = bldg.get("origin", [0, 0])
            width = bldg.get("width", 0)
            height = bldg.get("height", 0)
            self.grid[x:(x+width), y:(y+height)] = -2
            self.buildings.append([bldg['name'], x, y, width, height])


    
    def draw_buildings(self):
        ''' Draw the buildings '''
        rng = np.random.default_rng(self.seed)
        num_buildings = rng.choice(np.arange(2, MAX_BUILDINGS))
        while len(self.buildings) < num_buildings:
            # draw building from pool
            choice = rng.choice(np.arange(len(BUILDING_POOL)), p=BUILDING_POOL_DIST)
            bldg = BUILDING_POOL[choice]
            # choose coords - start with no constraints (i.e. a building can go anywhere)
            x, y = rng.choice(np.arange(10, self.grid_size[0] - 10), size=2)
            width = bldg['width']
            height = bldg['height']
            # check for overlap
            corners = [(x, y), (x, y+height), (x+width, y), (x+width, y+height)]
            
            # check for overlap with other buildings 
            has_overlap = False
            for b in self.buildings:
                for c_x, c_y in corners:
                    if ((b[1] - 10) <= c_x and c_x <= (b[1] + b[3] + 10)) and ((b[2] - 10) <= c_y and c_y <= (b[2] + b[4] + 10)):
                        has_overlap = True
                        break
            if not has_overlap:
                # mark terrain
                self.grid[x:(x+width),y:(y+height)] = -2
                # add building to internal array
                if bldg['limit'] == 1:
                    # if no more of a type can be drawn, remove from pool
                    BUILDING_POOL.pop(choice)
                    p = BUILDING_POOL_DIST.pop(choice)
                    BUILDING_POOL_DIST[0] += p
                else: 
                    bldg['limit'] -= 1
                self.buildings.append([bldg['name'], x, y, width, height])
            


    def init_fuel_grid(self, fuel_band):
        '''
        Creates fuel grid to store fuel types for all cells. 
        
        Fuel type is calculated using the following ranges for the NDVI measure:
        - (-1, 0] - Fuel #0 - Rocks, Water, etc.
        - (0, 0.3) - Fuel #1 - Sparse vegetation
        - [0.3, 0.6) - Fuel #2 - Shrubbery
        - [0.6, 1) - Fuel #3 - Dense vegetation
        '''
        sparse_veg = 1 * ((fuel_band > 0) & (fuel_band < 0.3))
        shrubbery = 2 * ((fuel_band >= 0.3) & (fuel_band < 0.6))
        dense_veg = 3 * (fuel_band >= 0.6)
        self.fuel_grid = sparse_veg + shrubbery + dense_veg
        self.grid[fuel_band <= 0] = -1
        self.non_fuel_mask = fuel_band <= 0
    
    def init_wind_grid(self, grid_size, generations=300):
        '''
        Creates the wind grid for this simulation

        Wind Grid - Stores the wind as a vector [u, v] in m/s
        '''
        res = [-1, -1, 2]
        for i in reversed(range(1, 9)):
            if grid_size[0] % i == 0 and res[0] == -1:
                res[0] = i
            if (grid_size[1]+generations) % i == 0 and res[1] == -1:
                res[1] = i
        self.noise = generate_perlin_noise_3d((grid_size[0], grid_size[1]+generations, 2), res)
        self.wind_grid = np.full(grid_size, self.wind) + (self.get_wind_adjustment())
    
    def get_wind_adjustment(self, offset=0):
        '''
        Randomly adjusts the wind. 
        '''
        return WIND_VARIABILITY * self.noise[:, offset:offset+self.grid_size[1], :] - (WIND_VARIABILITY / 2)
    
    def update_wind_grid(self, i=0):
        self.wind_grid = np.full((*self.grid_size, 2), self.wind) + self.get_wind_adjustment(i)

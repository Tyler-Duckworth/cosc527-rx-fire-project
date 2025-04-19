'''
COSC527 - Initial project experimentation

Simple file to test a very barebone fire simulation 

Based on this paper: https://www.publish.csiro.au/WF/WF11055
'''
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from osgeo import gdal
from perlin_numpy import generate_perlin_noise_3d

gdal.UseExceptions()

class RabbitQueue:
    def __init__(self):
        self._queue = deque()
    def put(self, item):
        self._queue.append(item)
    def get(self):
        return self._queue.popleft()

WIND_VARIABILITY = 2 # wind will be randomized with a magnitude of (-x, x)
WIND_PLOT_INTERVAL = 25 # controls the spacing between arrows on the wind plot
SHOW_ANIM = True
DEFAULT_AMBTEMP = 68
DEFAULT_RELHUM = 0.5
DEFAULT_WIND = "0,0"
DEFAULT_SMAP = 0.05
CELL_WIDTH = 30

# stores new rabbits to be added after a step is complete
rabbit_queue = RabbitQueue() 

'''
Fuel Lookup 
'''
FUEL_DATA = {
    # 0 - Can't burn (water, etc.)
    1: { # Sparse Vegetation
        "p_lifespan" : [0.5, 0.0, 0.0, 0.0], # probability that cell will reproduce for 1-n more days
        "f_h": 1, # fuel height (unitless)
        "v_g": 2.00, # pressure anomaly
        "C_h": 1.00, # isotropic hopping factor
        "C_w": 0.25, # wind weight
        "C_f": 0.0, # terrain fear factor
    },
    2: { # Shrubbery
        "p_lifespan" : [0.75, 0.50, 0.0, 0.0],
        "f_h": 1.5, 
        "v_g": 1.00, 
        "C_h": 1.00, 
        "C_w": 0.25,
        "C_f": 0.0
    },
    3: { # Dense vegetation
        "p_lifespan" : [1, 0.75, 0.25, 0.0],
        "f_h": 2, 
        "v_g": 1.00, 
        "C_h": 1.00, 
        "C_w": 0.25,
        "C_f": 0.0
    }
} 



class FireGrid:
    '''
    Raster Breakdown

    Bands:
    - 1 - NDVI (Greenness Index, [-1, 1])
    - 2 - Elevation (??, [0, 2000))
    - 3 - Slope (degrees, [0, 90])
    - 4 - Aspect (degrees, [0, 360])
    '''
    def __init__(self, input_file_name: str, generations: int):
        raw_file = gdal.Open(input_file_name)
        
        # background layer for plot
        self.bg = raw_file.GetRasterBand(1).ReadAsArray()
        self.generations = generations
        # Read in area metadata encoded into TIFF
        metadata = raw_file.GetMetadata()
        self.ambtemp = metadata.get("AMBTEMP", DEFAULT_AMBTEMP)
        self.relhum = metadata.get("RELHUM", DEFAULT_RELHUM)
        self.soil_moisture = metadata.get("SMAP", DEFAULT_SMAP)
        wind_arr = metadata.get("WIND", DEFAULT_WIND).split(",")
        self.wind = (float(wind_arr[0]), float(wind_arr[1]))
        self.grid_size = (raw_file.RasterXSize, raw_file.RasterYSize)
        
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
        
        self.create_fuel_grid(self.bg)
        self.init_fires([(y, x) for x in range(200, 250) for y in range(250, 260)])
        self.init_wind_grid((raw_file.RasterXSize, raw_file.RasterYSize, 2))
    
    def create_fuel_grid(self, fuel_band):
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

    def init_fires(self, fires):
        '''
        Initialized the various fires given an array of coordinates to light. 
        '''
        fire_mask = np.zeros(self.grid_size, dtype=bool)
        fire_mask[tuple(np.array(fires).T)] = True
        mask = ~self.non_fuel_mask & fire_mask
        self.grid[mask] = 1

    def init_wind_grid(self, grid_size, seed=None):
        '''
        Creates the wind grid for this simulation

        Wind Grid - Stores the wind as a vector [u, v] in m/s
        '''
        self.noise = generate_perlin_noise_3d((self.grid_size[0], self.grid_size[1]*2, 2), (3, 3, 2))
        self.wind_grid = np.full(grid_size, self.wind) + (self.get_wind_adjustment())
    
    def get_wind_adjustment(self, offset=0):
        '''
        Randomly adjusts the wind. 
        '''
        return WIND_VARIABILITY * self.noise[:, offset:offset+self.grid_size[1], :] - (WIND_VARIABILITY / 2)
    
    def calc_hop_distance(self, i, j, fuel_id):
        fuel = FUEL_DATA.get(fuel_id)
        if fuel is None:
            raise Exception(f"Undefined fuel type - ID: {fuel_id}, Coords: {(i, j)}")
        
        C_w = fuel.get("C_w", 0)
        C_h = fuel.get("C_h", 0)
        C_f = fuel.get("C_f", 0)
        z_r = 2 * (0.1 + np.random.random()) * fuel.get("f_h", 1)
        t = 2 * (2 * z_r / 9.81)**0.5  

        u = self.wind_grid[i, j, 0] / CELL_WIDTH
        v = self.wind_grid[i, j, 1] / CELL_WIDTH
        s = self.slope_grid[i, j]
        dx = (C_w * u * abs(u) + (10 * C_f * s * abs(s))) * t + C_h * z_r * (0.5 - np.random.random())
        dy = (C_w * v * abs(v)  + (10 * C_f * s * abs(s))) * t + C_h * z_r * (0.5 - np.random.random())

        return dx, dy

    def update_grid(self, frame_no: int):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.advance_grid_cell(i, j)
        while len(rabbit_queue._queue) > 0:
            i, j = rabbit_queue.get()
            self.grid[i, j] = 1
            self.lifespan_grid[i, j] = 1
        # update wind grid
        self.wind_grid = np.full((*self.grid_size, 2), self.wind) + self.get_wind_adjustment(frame_no)
    
    def create_child(self, i, j):
        fuel_type = self.fuel_grid[i, j]
        if fuel_type == 0: 
            return
        
        dx, dy = self.calc_hop_distance(i, j, fuel_type)
        new_i = int(i + dx)
        new_j = int(j + dy)

        # add to queue 
        if 0 <= new_i < self.grid_size[0] and 0 <= new_j < self.grid_size[1]:
            if self.grid[new_i, new_j] == 0:
                rabbit_queue.put((new_i, new_j))  

    def advance_grid_cell(self, i, j):
        val = self.grid[i, j]
        if val == 1: # Burning - Produce children, set to dying
            num_children = np.random.randint(1, 6)
            for k in range(num_children):
                self.create_child(i, j)
            # roll dice - see if it survives this day
            if np.random.random() <= FUEL_DATA.get(self.fuel_grid[i, j])['p_lifespan'][self.lifespan_grid[i, j] - 1]:
                self.lifespan_grid[i, j] += 1
            else:
                self.grid[i, j] = 2
        elif val == 2: # dying - mark cell as dead
            self.grid[i, j] = -1
    def plot_wind(self, ax):
        ax.quiver(np.arange(0, self.grid_size[1], 10), np.arange(0, self.grid_size[1], 10), self.wind_grid[::10, ::10, 0], self.wind_grid[::10, ::10, 1])
        return ax


def run_simulation(input_file_name="", generations=100):
    grid = FireGrid(input_file_name, generations)
    cmap = colors.ListedColormap(['black', 'green', 'orange', 'red'])
    norm = colors.BoundaryNorm([-1, 0, 1, 2, 3], cmap.N)   
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6)) 

    im1 = axs[0].imshow(grid.bg, cmap='binary', vmin=-1, vmax=1)
    im2 = axs[0].imshow(grid.grid, alpha=.5, cmap=cmap, norm=norm, animated=True)
    axs[0].set_title('Fire Spread')
    
    axs[1].imshow(grid.fuel_grid, cmap=cmap, norm=norm)
    axs[1].set_title("Fuel Distribution")

    wind_ticks = np.arange(0, grid.grid_size[1], WIND_PLOT_INTERVAL)
    im3 = axs[2].quiver(wind_ticks, wind_ticks, grid.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 0], grid.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 1])
    axs[2].set_title("Wind Grid")
    def init(): 
        """ Dummy function (FuncAnimation runs this once before plotting) """
        return im1, im2, im3

    def update_anim(frame_no):
        grid.update_grid(frame_no)
        im2.set_array(grid.grid)
        im3.set_UVC(grid.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 0], grid.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 1])
        return im1, im2, im3

    anim = animation.FuncAnimation(fig, update_anim, init_func=init, frames=generations, blit=True, repeat=True) 
    plt.tight_layout()
    if SHOW_ANIM:
        plt.show()    
    else:   
        anim.save("demo.mp4")

if __name__ == "__main__":
    run_simulation("data/stacked_ca.tif")
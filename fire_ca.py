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
import matplotlib.patches as patches
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

MAX_BUILDINGS = 10
BUILDING_POOL_DIST = [0.5, 0.25, 0.2, 0.05]
BUILDING_POOL = [
    {"width": 5, "height": 7, "name": "Research Station", "limit": 20},
    {"width": 2, "height": 2, "name": "Fire Tower", "limit": 5},
    {"width": 2, "height": 3, "name": "Bathroom", "limit": 10},
    {"width": 30, "height": 30, "name": "Town", "limit": 1}
]

class Environment:
    ''' Represents the simulation environment '''
    def __init__(self, source_file: str):
        raw_file = gdal.Open(source_file)
        self.grid_size = (raw_file.RasterXSize, raw_file.RasterYSize)
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
        
        self.init_fuel_grid(self.bg)
        self.init_wind_grid((raw_file.RasterXSize, raw_file.RasterYSize, 2))
        self.buildings = []

    def draw_buildings(self):
        ''' Draw the buildings '''
        num_buildings = np.random.choice(np.arange(2, MAX_BUILDINGS))
        for i in range(num_buildings):
            # draw building from pool
            choice = np.random.choice(np.arange(len(BUILDING_POOL)), p=BUILDING_POOL_DIST)
            bldg = BUILDING_POOL[choice]
            if bldg['limit'] == 1:
                # if no more of a type can be drawn, remove from pool
                BUILDING_POOL.pop(choice)
                p = BUILDING_POOL_DIST.pop(choice)
                BUILDING_POOL_DIST[0] += p
            else: 
                bldg['limit'] -= 1
            # choose coords - start with no constraints (i.e. a building can go anywhere)
            x, y = np.random.choice(np.arange(self.grid_size[0]), size=2)
            width = bldg['width']
            height = bldg['height']
            # mark terrain
            self.grid[x:(x+width),y:(y+height)] = -2
            # add building to internal array
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
    
    def update_wind_grid(self, i=0):
        self.wind_grid = np.full((*self.grid_size, 2), self.wind) + self.get_wind_adjustment(i)

class FireSimulation:
    '''
    Raster Breakdown

    Bands:
    - 1 - NDVI (Greenness Index, [-1, 1])
    - 2 - Elevation (??, [0, 2000))
    - 3 - Slope (degrees, [0, 90])
    - 4 - Aspect (degrees, [0, 360])
    '''
    def __init__(self, environment: Environment, generations: int):
        self.env = environment
        self.generations = generations
    

    def init_fires(self, fires):
        '''
        Initialized the various fires given an array of coordinates to light. 
        '''
        fire_mask = np.zeros(self.grid_size, dtype=bool)
        fire_mask[tuple(np.array(fires).T)] = True
        mask = ~self.non_fuel_mask & fire_mask
        self.grid[mask] = 1
    
    def calc_hop_distance(self, i, j, fuel_id):
        fuel = FUEL_DATA.get(fuel_id)
        if fuel is None:
            raise Exception(f"Undefined fuel type - ID: {fuel_id}, Coords: {(i, j)}")
        
        C_w = fuel.get("C_w", 0)
        C_h = fuel.get("C_h", 0)
        C_f = fuel.get("C_f", 0)
        z_r = 2 * (0.1 + np.random.random()) * fuel.get("f_h", 1)
        t = 2 * (2 * z_r / 9.81)**0.5  

        u = self.env.wind_grid[i, j, 0] / CELL_WIDTH
        v = self.env.wind_grid[i, j, 1] / CELL_WIDTH
        s = self.env.slope_grid[i, j]
        dx = (C_w * u * abs(u) + (10 * C_f * s * abs(s))) * t + C_h * z_r * (0.5 - np.random.random())
        dy = (C_w * v * abs(v)  + (10 * C_f * s * abs(s))) * t + C_h * z_r * (0.5 - np.random.random())

        return dx, dy

    def update_grid(self, frame_no: int):
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                self.advance_grid_cell(i, j)
        while len(rabbit_queue._queue) > 0:
            i, j = rabbit_queue.get()
            self.env.grid[i, j] = 1
            self.env.lifespan_grid[i, j] = 1
        self.env.update_wind_grid(frame_no)
        
    
    def create_child(self, i, j):
        fuel_type = self.env.fuel_grid[i, j]
        if fuel_type == 0: 
            return
        
        dx, dy = self.calc_hop_distance(i, j, fuel_type)
        new_i = int(i + dx)
        new_j = int(j + dy)

        # add to queue 
        if 0 <= new_i < self.env.grid_size[0] and 0 <= new_j < self.env.grid_size[1]:
            if self.env.grid[new_i, new_j] == 0:
                rabbit_queue.put((new_i, new_j))  

    def advance_grid_cell(self, i, j):
        val = self.env.grid[i, j]
        if val == 1: # Burning - Produce children, set to dying
            num_children = np.random.randint(1, 6)
            for k in range(num_children):
                self.create_child(i, j)
            # roll dice - see if it survives this day
            if np.random.random() <= FUEL_DATA.get(self.env.fuel_grid[i, j])['p_lifespan'][self.env.lifespan_grid[i, j] - 1]:
                self.env.lifespan_grid[i, j] += 1
            else:
                self.env.grid[i, j] = 2
        elif val == 2: # dying - mark cell as dead
            self.grid[i, j] = -1


def run_simulation(input_file_name="", generations=100):
    env = Environment(input_file_name)
    env.draw_buildings()
    sim = FireSimulation(env, generations)
    cmap = colors.ListedColormap(['blue', 'black', 'green', 'orange', 'red'])
    norm = colors.BoundaryNorm([-2, -1, 0, 1, 2, 3], cmap.N)   
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(24,6)) 

    im1 = axs[0].imshow(env.bg, cmap='binary', vmin=-1, vmax=1)
    im2 = axs[0].imshow(env.grid, alpha=.5, cmap=cmap, norm=norm, animated=True)
    patch_list = []
    for bldg in env.buildings:
        patch_list.append(axs[0].add_patch(patches.Rectangle((bldg[2], bldg[1]), bldg[4], bldg[3], linewidth=1, edgecolor='r')))
    axs[0].set_title('Fire Spread')
    

    axs[1].imshow(env.fuel_grid, cmap=cmap, norm=norm)
    axs[1].set_title("Fuel Distribution")

    wind_ticks = np.arange(0, env.grid_size[1], WIND_PLOT_INTERVAL)
    im3 = axs[2].quiver(wind_ticks, wind_ticks, env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 0], env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 1])
    axs[2].set_title("Wind Grid")

    elevation = axs[3].imshow(env.elevation, cmap="terrain", vmin=0, vmax=1)
    fig.colorbar(elevation, ax=axs[3])
    def init():
        """ Dummy function (FuncAnimation runs this once before plotting) """
        return im1, im2, im3, *patch_list

    def update_anim(frame_no):
        sim.update_grid(frame_no)
        im2.set_array(env.grid)
        im3.set_UVC(env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 0], env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 1])
        
        return im1, im2, im3, *patch_list

    anim = animation.FuncAnimation(fig, update_anim, init_func=init, frames=generations, blit=True, repeat=True) 
    plt.tight_layout()
    if SHOW_ANIM:
        plt.show()    
    else:   
        anim.save("demo.mp4")

if __name__ == "__main__":
    run_simulation("data/stacked_ca.tif")
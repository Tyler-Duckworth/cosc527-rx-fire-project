'''
Prescribed Fire Simulation

By: Tyler Duckworth, Andy Zheng, and Gabriel Laboy

Based on this paper: https://www.publish.csiro.au/WF/WF11055
'''
import argparse
import os
import time
import sys
import json
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from osgeo import gdal
import matplotlib.patches as patches

from environment import Environment
from utils import RabbitQueue
gdal.UseExceptions()



WIND_PLOT_INTERVAL = 25 # controls the spacing between arrows on the wind plot
SHOW_ANIM = True
OUTPUT_DIR = "results"
OUTPUT_FILE = "anim.mp4"
CELL_WIDTH = 30
GENERATIONS = 300
SWITCH_FRAME = 25
'''
Fuel Lookup 
'''
FUEL_DATA = {
    # 0 - Can't burn (water, etc.)
    1: { # Sparse Vegetation
        "p_lifespan" : [0.5, 0.0, 0.0, 0.0], # probability that cell will reproduce for 1-n more days
        "f_h": 1, # fuel height (unitless)
        "v_g": 2.00, # pressure anomaly
        "C_h": 0.25, # isotropic hopping factor
        "C_w": 0.25, # wind weight
        "C_f": 0.0, # terrain fear factor
    },
    2: { # Shrubbery
        "p_lifespan" : [0.75, 0.50, 0.0, 0.0],
        "f_h": 1.5, 
        "v_g": 1.00, 
        "C_h": 0.5, 
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


class FireSimulation:
    '''
    Contains logic to update the simulation, spawn children, and add fires to the environment
    '''
    def __init__(self, environment: Environment, generations: int):
        self.env = environment
        self.generations = generations
        # stores new rabbits to be added after a step is complete
        self.rabbit_queue = RabbitQueue() 

    def init_fires(self, fires):
        '''
        Initialized the various fires given an array of coordinates to light. 
        '''
        fire_mask = np.zeros(self.env.grid_size, dtype=bool)
        fire_mask[tuple(np.array(fires).T)] = True
        mask = ~self.env.non_fuel_mask & fire_mask
        self.env.grid[mask] = 1
    
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
        s = 0 # disabled until slope can be fixed
        dx = (C_w * u * abs(u) + (10 * C_f * s * abs(s))) * t + C_h * z_r * (0.5 - np.random.random())
        dy = (C_w * v * abs(v)  + (10 * C_f * s * abs(s))) * t + C_h * z_r * (0.5 - np.random.random())

        return dx, dy

    def update_grid(self, frame_no: int):
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                self.advance_grid_cell(i, j)
        while len(self.rabbit_queue._queue) > 0:
            i, j = self.rabbit_queue.get()
            self.env.grid[i, j] = 1
            self.env.lifespan_grid[i, j] = 1
        self.env.update_wind_grid(frame_no)
        self.env.save(frame_no)
    
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
                self.rabbit_queue.put((new_i, new_j))  

    def advance_grid_cell(self, i, j):
        '''
        Updates a particular cell depending on its state
        '''
        val = self.env.grid[i, j]
        if val == 1: # Burning - Produce children
            num_children = np.random.randint(1, 6)
            for k in range(num_children):
                self.create_child(i, j)
            # lifespan check - see if this cell survives this day
            if np.random.random() <= FUEL_DATA.get(self.env.fuel_grid[i, j])['p_lifespan'][self.env.lifespan_grid[i, j] - 1]:
                self.env.lifespan_grid[i, j] += 1
            else:
                self.env.grid[i, j] = 2 # cell is dying (dead by next generation)
        elif val == 2: # dying - mark cell as dead
            self.env.grid[i, j] = -1

    def create_firebreaks(self):
        ''' Draw firebreaks around the buildings '''
        FIREBREAK_SIZE = 2
        GAP_SIZE = 5
        ignition_points = []
        for bldg in self.env.buildings:
            # inner firebreak - Inner edge to protect the building'
            x0 = bldg[1] - 1
            x1 = bldg[1] + bldg[3]
            y0 = bldg[2] - 1
            y1 = bldg[2] + bldg[4]
            x_range = np.arange(x0, x1)
            y_range = np.arange(y0, y1)
            self.env.grid[x0,  y0:y1] = -2
            self.env.grid[(x1-1), y0:y1] = -2
            self.env.grid[x0:x1, [y0] * (x1-x0+1)] = -2
            self.env.grid[x0:x1, [y1-1] * (x1-x0+1)] = -2
            # (x0, y1)                       (x1, y1)
            # ------------------------------
            # |                             | 
            # |                             | 
            # |                             | 
            # |                             | 
            # ------------------------------
            # (x0, y0)                       (x1, y0)
            
            x0 = bldg[1] - GAP_SIZE - 1
            x1 = bldg[1] + bldg[3] + FIREBREAK_SIZE + GAP_SIZE
            y0 = bldg[2] - FIREBREAK_SIZE - GAP_SIZE
            y1 = bldg[2] + bldg[4] + GAP_SIZE
            x_range = np.arange(x0, x1+1)
            y_range = np.arange(y0, y1+1)
            self.env.grid[x0-FIREBREAK_SIZE:x0, y0-(FIREBREAK_SIZE-1):y1+FIREBREAK_SIZE] = -3 # top 
            self.env.grid[x1-FIREBREAK_SIZE:x1, y0:y1] = -3 # bottom
            self.env.grid[x0:x1, y0-FIREBREAK_SIZE+1:y0+1] = -3 # left
            self.env.grid[x0:x1, y1:y1+FIREBREAK_SIZE] = -3 # right

            # set bottom and right edges on fire (TODO: Add logic to pick the upwind edges)
            ignition_points += (list(zip([x1-FIREBREAK_SIZE-1] * (y1-y0), y_range[1:-1])))
            ignition_points += (list(zip(x_range[0:-3], [y1-1] * (x1-x0))))
        self.init_fires(ignition_points) # Add fires for prescribed burn phase

def run_simulation(input_file_name="", seed=500, config_file_name=None):
    env = Environment(input_file_name, OUTPUT_DIR, seed, GENERATIONS)
    if config_file_name is None:
        env.draw_buildings()
    else:
        env.load_buildings_from_file(config_file_name)
    sim = FireSimulation(env, GENERATIONS)
    sim.create_firebreaks()
    
    cmap = colors.ListedColormap(['blue', 'black', 'green', 'orange', 'red'])
    norm = colors.BoundaryNorm([-2, -1, 0, 1, 2, 3], cmap.N)   

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(env.fuel_grid, cmap='terrain')
    ax.set_title("Fuel Distribution")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(f"{OUTPUT_DIR}/fuel_distribution.png", bbox_inches='tight')
    fig.savefig(f"{OUTPUT_DIR}/fuel_distribution.svg", bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6)) 

    # Primary Plot - Show state grid with background map layer
    im1 = axs[0].imshow(env.bg, cmap='binary', vmin=-1, vmax=1)
    im2 = axs[0].imshow(env.grid, alpha=.5, cmap=cmap, norm=norm, animated=True)
    time_text = axs[0].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', c='yellow', transform=axs[0].transAxes)
    patch_list = []
    for bldg in env.buildings:
        patch_list.append(axs[0].add_patch(patches.Rectangle((bldg[2]-1, bldg[1]-1), bldg[4], bldg[3], linewidth=1, edgecolor='r')))
    axs[0].set_title('Fire Spread')
    
    wind_ticks = np.arange(0, env.grid_size[1], WIND_PLOT_INTERVAL)
    im3 = axs[1].quiver(wind_ticks, wind_ticks, env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 0], env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 1])
    axs[1].set_title("Wind Grid")

    def init():
        """ Dummy function (FuncAnimation runs this once before plotting) """
        time_text.set_text("Init")
        return im1, im2, im3, time_text, *patch_list

    def update_anim(frame_no):
        if frame_no == SWITCH_FRAME:
            # switch phases - add fire to random part of map
            sim.init_fires([(x, y) for x in range(300, 360) for y in range(300, 360)])
        sim.update_grid(frame_no)
        im2.set_array(env.grid)
        im3.set_UVC(env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 0], env.wind_grid[::WIND_PLOT_INTERVAL, ::WIND_PLOT_INTERVAL, 1])
        time_text.set_text(frame_no)
        return im1, im2, im3, time_text, *patch_list

    anim = animation.FuncAnimation(fig, update_anim, init_func=init, frames=GENERATIONS, blit=True, repeat=False) 
    plt.tight_layout()
    if SHOW_ANIM:
        plt.show()
    print("Saving animation...") 
   
    anim.save(f"{OUTPUT_DIR}/{OUTPUT_FILE}", progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'), dpi=300)
    print("Finished saving animation...")    

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="fire_simulation", 
        description="Simulates a prescribed fire using a generated GeoTIFF file. See file for more info."
    )
    parser.add_argument("data_file", nargs="?", default="data/stacked_ca.tif", help="Path to .tif file to run the simulation on")
    parser.add_argument("-c", "--config-file", type=str, help="Path to pre-built config file (if using)")
    parser.add_argument("--hide-animation", help="Hides the animation", action="store_true")
    parser.add_argument("-d", "--output-dir", type=str, help="Output directory to save intermediate data (defaults to \"results\")", default="results")
    parser.add_argument("-o", "--output-file", type=str, help="Name of file to save the animation to as an MP4 (defaults to \"[tn|ca]_anim.mp4\")", default="anim.mp4")
    parser.add_argument("-g", "--generations", type=int, help="Total number of generations to run the simulation over. (Defaults to 200)", default=200)
    parser.add_argument("-f", "--switch-frame", type=int, help="Specifies which frame to switch from prescribed fires (Phase 1) to the forest fire (Phase 2). (Defaults to 25)", default=25)
    parser.add_argument("-s", "--seed", type=int, help="Random number seed", default=int(time.time()))
    args = parser.parse_args(sys.argv[1:])
    return args



if __name__ == "__main__":
    args = parse_arguments()
    try:
        SHOW_ANIM = not args.hide_animation
        OUTPUT_DIR = args.output_dir
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        OUTPUT_FILE = args.output_file
        GENERATIONS = args.generations
        SWITCH_FRAME = args.switch_frame
        run_simulation(args.data_file, args.seed, args.config_file)
    except Exception as e:
        print("ERROR")
        print(e)
        raise e
        exit(1)
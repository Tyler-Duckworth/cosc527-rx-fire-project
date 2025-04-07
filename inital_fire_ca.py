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
import opensimplex


class RabbitQueue:
    def __init__(self):
        self._queue = deque()
    def put(self, item):
        self._queue.append(item)
    def get(self):
        return self._queue.popleft()

global grid_size # shared grid size 
'''
Main grid - represents all the different cell's states
Grid states:
- -1 - Burned (fuel eaten)
-  0 - Unburned 
-  1 - Burning (reproducing)
-  2 - Dying
'''
global grid

'''
Landscape grid - Stores the fuel type (if any) of each one)
'''
global fuel_grid 

global fuel_data # dict of the different fuels available
global rabbit_queue # stores new rabbits to be added after a step is complete


# TODO: Add in logic to adjust the pressure and wind? 
global wind
global pressure_grid


rabbit_queue = RabbitQueue()

wind = (-3, 2)
fuel_data = {
    1: {"m1": 0.5, "m2": 0.0, "m3": 0.0, "m4": 0.0, "m5": 0.0,
           "f_h": 0.65, "v_g": 2.00, "C_h": 1.00, "C_w": 0.25,
           "C_f": 0.0, "delta_B": 1},
    2: {"m1": 0.1, "m2": 0.0, "m3": 0.0, "m4": 0.0, "m5": 0.0,
           "f_h": 1.00, "v_g": 1.00, "C_h": 1.00, "C_w": 0.25,
           "C_f": 0.0, "delta_B": 1}
}


def init_grid(grid_size, _type):
    return np.zeros(grid_size, dtype=_type)

def init_landscape(grid_size):
    global fuel_grid
    fuel_grid = np.ones(grid_size)

def init_fires(fires):
    for i, j in fires:
        grid[i, j] = 1

def calc_hop_distance(fuel_id):
    fuel = fuel_data.get(fuel_id)
    if fuel is None:
        return 0, 0
    
    C_w = fuel.get("C_w", 0)
    C_h = fuel.get("C_h", 0)
    z_r = 2 * (0.1 + np.random.random()) * fuel.get("f_h", 1)
    t = 2 * (2 * z_r / 9.81)**0.5

    # Equation 1 
    dx = (C_w * wind[0] * abs(wind[0]) ) * t + C_h * z_r * (0.5 - np.random.random())
    dy = (C_w * wind[1] * abs(wind[1]) ) * t + C_h * z_r * (0.5 - np.random.random())

    return dx, dy

def create_child(i, j):
    fuel_type = fuel_grid[i, j]
    if fuel_type == 0: 
        return
    dx, dy = calc_hop_distance(fuel_type)
    new_i = int(i + dx)
    new_j = int(j + dy)

    if 0 <= new_i < grid_size[0] and 0 <= new_j < grid_size[1]:
        if grid[new_i, new_j] == 0:
            rabbit_queue.put((new_i, new_j))
    

def advance_grid_cell(i, j):
    val = grid[i, j]
    if val == 1: # Burning - Produce children, set to dying
        num_children = np.random.randint(1, 6)
        for k in range(num_children):
            create_child(i, j)
        grid[i, j] = 2
    elif val == 2: # dying - mark cell as dead
        grid[i, j] = -1
    

def update_grid():
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            advance_grid_cell(i, j)
    while len(rabbit_queue._queue) > 0:
        i, j = rabbit_queue.get()
        grid[i, j] = 1


if __name__ == "__main__":
    nrows = 300
    ncols = 300
    grid_size = (nrows, ncols)
    generations = 100
    
    init_landscape(grid_size)
    grid = init_grid(grid_size, int)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if fuel_grid[i, j] == 0: 
                grid[i, j] = -1
    pressure_grid = init_grid(grid_size, float)
    init_fires([(y, x) for x in range(75, 225) for y in range(200, 210)])

    cmap = colors.ListedColormap(['black', 'green', 'orange', 'red'])
    norm = colors.BoundaryNorm([-1, 0, 1, 2, 3], cmap.N)   
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
    im1 = axs[0].imshow(grid, cmap=cmap, norm=norm, animated=True)
    axs[0].set_title('Fire Spread')
    im2 = axs[1].imshow(pressure_grid, cmap='plasma', animated=True)  # Use 'plasma' colormap
    axs[1].set_title('Pressure Anomalies')
    
    def init(): 
        """ Dummy function (FuncAnimation runs this once before plotting) """
        return im1, im2, 
    
    def update_anim(frame):
        update_grid()
        im1.set_array(grid)
        im2.set_array(pressure_grid)
        return im1, im2,

    anim = animation.FuncAnimation(fig, update_anim, init_func=init, frames=generations, blit=True) 
    plt.tight_layout()
    anim.save("demo.mp4")
    
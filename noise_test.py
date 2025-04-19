from perlin_numpy import generate_perlin_noise_2d, generate_perlin_noise_3d
import numpy as np
import sys
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
noise = generate_perlin_noise_2d((300, 600), (5, 5))

def animate(i):
    plt.clf()
    plt.imshow(noise[0:300, i:(i+300)], cmap='gray')
    return plt

fig = plt.figure()

anim = FuncAnimation(fig, animate, frames=300, interval=50, blit=False)
plt.show()
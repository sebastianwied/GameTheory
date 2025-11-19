import numpy as np
from Perlin_numpy import *
import pandas as pd
import matplotlib.pyplot as plt

def gaussian3D(xSig, ySig, amp, width, height, gridN, truncate=100, norm=False):
    xR = np.linspace(-width/2, width/2, gridN)
    yR = np.linspace(-height/2, height/2, gridN)
    #print(xR.shape, yR.shape)
    def value(x, y):
        return amp*np.exp((-x*x/(2*xSig*xSig))+(-y*y/(2*ySig*ySig)))
    grid = np.zeros((gridN, gridN))
    for idx in range(gridN):
        for idy in range(gridN):
            #print(idx, idy)
            grid[idy, idx] = min(value(xR[idx], yR[-(idy+1)]), truncate)
    if norm: grid /= np.nanmax(grid)
    return xR, yR, grid

xR, yR, grid = gaussian3D(1,1,.1,16,16,128, truncate = 2, norm=False)
X, Y = np.meshgrid(xR, yR)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, grid)
plt.imshow(grid, cmap="viridis")
plt.colorbar()
plt.show()

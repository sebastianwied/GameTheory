from memoryN import *
from Perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d
import numpy.random as random

def randomGrid(N, maxN=1):
    grid = [[Rand(1,1,maxN) for _ in range(N)] for _ in range(N)]
    return np.array(grid)

def blankGrid(N, res, maxN=1, seed=None):
    if seed == None:
        seed = int(random.rand()*1000)
    rng = np.random.default_rng(seed)
    print(seed)
    paramMaps = generate_fractal_noise_2d((N,N), res, seed=seed, number=(4**maxN))
    paramMaps += np.ones(paramMaps.shape)*1.3
    paramMaps /= 2
    grid = np.array([[BLANK(1,1,maxN) for _ in range(N)] for _ in range(N)])
    for idr in range(N):
        for idc in range(N):
            grid[idr,idc].setRule([i**2 for i in list(paramMaps[:,idr,idc])])
    return np.array(grid)
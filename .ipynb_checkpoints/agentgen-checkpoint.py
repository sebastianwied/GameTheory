import numpy as np
import numpy.random as random
from params import *
from Perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d

def perlinGrid(shape, res, seed=None):
    if seed == None:
        seed = int(random.rand()*1000)
    rng = np.random.default_rng(seed)
    print(seed)
    agents = list(agentTypes.values())
    rng.shuffle(agents)
    noise = generate_fractal_noise_2d(shape, res, seed=seed)
    noise *= len(agents)
    noise += len(agents)
    noise /= 2
    noise = np.floor(noise).astype(int)
    agentgrid = [[] for _ in range(shape[0])]
    for idr, row in enumerate(noise):
        for item in row:
            agentgrid[idr].append(agents[item][0]())
    agentgrid = np.array(agentgrid)
    return agentgrid

def noiseGrid(shape):
    agents = list(agentTypes.values())
    print(agents)
    grid = np.floor(random.rand(Y,X)*(len(agents))).astype(int)
    agentgrid = [[] for _ in range(Y)]
    for idr, row in enumerate(grid):
        for item in row:
            agentgrid[idr].append(agents[item]())
    agentgrid = np.array(agentgrid)
    return agentgrid
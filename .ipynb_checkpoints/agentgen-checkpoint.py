import numpy as np
import numpy.random as random
from agents import *
from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d

def perlinGrid(shape, res):
    agents = list(agentTypes.values())
    print(agents)
    random.shuffle(agents)
    noise = generate_fractal_noise_2d(shape, res)
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
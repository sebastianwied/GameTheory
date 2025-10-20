import numpy as np
import numpy.random as random
from params import *
from Perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d
import memoryone as m1

def perlinGrid(shape, res, seed=None):
    if seed == None:
        seed = int(random.rand()*1000)
    rng = np.random.default_rng(seed)
    print(seed)
    agents = list(agentTypes.values())
    rng.shuffle(agents)
    noise = generate_fractal_noise_2d(shape, res, seed=seed)[0]
    noise += 1
    noise /= 2
    noise *= len(agents)
    noise = np.floor(noise).astype(int)
    agentgrid = [[] for _ in range(shape[0])]
    for idr, row in enumerate(noise):
        for item in row:
            agentgrid[idr].append(agents[item][0]())
    agentgrid = np.array(agentgrid)
    return agentgrid

def noiseParamGrid(shape):
    agentgrid = np.full(shape, m1.BLANK)
    for idr in range(shape[0]):
        for idc in range(shape[1]):
            agentgrid[idr,idc] = m1.BLANK()
            agentgrid[idr,idc].rule = list(random.rand(4))
    return agentgrid

def perlinParamGrid(shape, res, seed=None):
    if seed == None:
        seed = int(random.rand()*1000)
    rng = np.random.default_rng(seed)
    print(seed)
    paramMaps = (generate_fractal_noise_2d(shape, res, seed=seed, number=4)+1)/2
    print(paramMaps.shape)
    agentgrid = np.full(shape, m1.BLANK)
    for idr in range(shape[0]):
        for idc in range(shape[1]):
            agentgrid[idr,idc] = m1.BLANK()
            agentgrid[idr,idc].rule = [i**2 for i in list(paramMaps[:,idr,idc])]
    return agentgrid
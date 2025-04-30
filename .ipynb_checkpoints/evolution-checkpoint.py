import numpy as np
import numpy.random as random
from memoryone import *

def gridBasicEvolution(agents, adjacencies, scoresGained):
    Y, X = agents.shape
    agentscopy = np.copy(agents)
    for idr, row in enumerate(agents):
        for idc, agent in enumerate(row):
            neighbors = adjacencies[idr][idc]
            neighborscores = [scoresGained[neighbor // Y, neighbor % X] for neighbor in neighbors]
            maxscore = max(neighborscores)
            index = neighbors[neighborscores.index(maxscore)]
            if (random.rand() < .5) and scoresGained[idr, idc] < maxscore:
                agentscopy[idr, idc] = type(agents[index // Y, index % X])(score=scoresGained[idr, idc])
    return agentscopy

def gridNoEvolution(agents, adjacencies, scoresGained):
    return agents
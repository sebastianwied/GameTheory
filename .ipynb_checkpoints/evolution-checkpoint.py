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

def gridParameterEvolution(agents, adjacencies, scoresGained):
    '''
    Input: Memory-one agents
    Goal: Evolve the 4 parameters of each agent.
    How: Each agent looks at its highest scoring neighbor and finds the parameter with the largest difference.
    Adjusts it's own ruleset to more closely match.
    Ex:
    Top: [0.7, 0.6, 0.9, 0], Mine: [0.4, 0.3, 0.9, .9]
    Diff = [0.3, 0.3, 0, -0.9], squared = [0.09, 0.09, 0, 0.81],
    Rule change: [0, 0, 0, -0.81]
    '''
    Y, X = agents.shape
    agentscopy = np.copy(agents)
    for idr, row in enumerate(agents):
        for idc, agent in enumerate(row):
            neighbors = adjacencies[idr][idc]
            neighborscores = [scoresGained[neighbor // Y, neighbor % X] for neighbor in neighbors]
            maxscore = max(neighborscores)
            index = neighbors[neighborscores.index(maxscore)]
            if scoresGained[idr, idc] < maxscore:
                topRule = agents[index // Y, index % X].rule
                myRule = agent.rule
                ruleDiff = [(topRule[i] - myRule[i]) for i in range(len(topRule))]
                ruleDiffSq = [np.sign(ruleDiff[i])*(ruleDiff[i])**2 for i in range(len(topRule))]
                maxChangeId = np.argmax(np.abs(ruleDiff))
                ruleDelta = [0 if i != maxChangeId else ruleDiff[i] for i in range(len(topRule))]
                agentscopy[idr, idc].rule = [myRule[i] + ruleDelta[i] for i in range(len(topRule))]
    return agentscopy
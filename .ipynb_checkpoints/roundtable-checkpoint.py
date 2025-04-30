import numpy as np
import numpy.random as random
import math
from params import *

def distributePrize(me, them):
    if me == them == COOP:
        return payoff[1]
    if (me == COOP) and (them == DEF):
        return payoff[3]
    if (me == DEF) and (them == COOP):
        return payoff[0]
    return payoff[2]

def playOneIteration(agent1, agent2, roundNumber, error=None):
    '''
    Inputs: Two agent objects, iteration round number
    Output: Tuple of scores given to each player
    '''
    move1 = agent1.rule(agent2.past, roundNumber)
    move2 = agent2.rule(agent1.past, roundNumber)
    if error != None:
        if np.random.rand() < error:
            move1 = 1 - move1
        if np.random.rand() < error:
            move2 = 1 - move2
    agent1.past.append(move1)
    agent2.past.append(move2)
    a1prize = distributePrize(move1, move2)
    a2prize = distributePrize(move2, move1)
    return a1prize, a2prize

def playNIterations(agent1, agent2, N, error=None):
    '''
    Inputs: Two agent objects, number of iterations
    Outputs: Tuple of scores accumulated for each player
    '''
    a1score = 0
    a2score = 0
    for n in range(N):
        a1p, a2p = playOneIteration(agent1, agent2, n, error)
        a1score += a1p
        a2score += a2p
    agent1.past = []
    agent2.past = []
    return a1score, a2score

'''
Defines all the adjacency table generation and round generation methods.
'''

def gridAdjacencyTable(agentGrid, neighborhoodSize: int):
    '''
    Inputs:
    neighborhoodSize: integer defining size of the moore neigborhood.
    agentGrid: numpy 2d array of agents.
    Outputs:
    list of neighbors for each index
    List of neighbors stored just as the agent number - 0 is the top left. 
    y coord is agentNumber // # of columns
    x coord is agentNumber % # of rows
    '''
    Y, X = agentGrid.shape
    adjtable = [[0 for _ in range(X)] for _ in range(Y)]
    for y in range(Y):
        for x in range(X):
            # Agent (x,y). [xMin, xMax), [yMin, yMax)
            xMin, xMax = (max(x-neighborhoodSize, 0), min(x+neighborhoodSize, X-1))
            yMin, yMax = (max(y-neighborhoodSize, 0), min(y+neighborhoodSize, Y-1))
            size = (xMax+1-xMin)*(yMax+1-yMin) - 1
            neighbors = []
            for yn in range(yMin, yMax + 1):
                for xn in range(xMin, xMax + 1):
                    if (xn == x) and (yn == y): continue
                    neighbors.append(yn*Y + xn)
            adjtable[y][x] = neighbors
    return adjtable

def gridGenerateRound(adjtable, agentGrid):
    '''
    Generate list of 2-tuples (player1, player2) stored as agentNumber
    Inputs:
    adjtable: 2d numpy grid output matching the output from gridAdjacencyTable
    agentGrid: numpy 2d array of agents.
    '''
    Y, X = agentGrid.shape
    N = X*Y
    placed = np.zeros((Y,X))
    playersleft = N
    games = []
    while playersleft > 1:
        player = int(random.rand()*N)
        while placed[player//Y][player%Y] != 0:
            player = int(random.rand()*N)
        playersleft -= 1
        placed[player//Y][player%X] = 1
        neighbors = adjtable[player//Y][player%Y]
        weights = np.ones(len(neighbors))
        for ind, neighbor in enumerate(neighbors): 
            weights[ind] -= placed[neighbor//Y][neighbor%Y]
        if np.sum(weights) == 0: continue
        weights /= np.sum(weights)
        opponent = random.choice(neighbors, p=weights)
        playersleft -= 1
        placed[opponent//Y][opponent%Y] = 1
        games.append((player, opponent))
        #print(games[-1], placed)
    return games

def gridNumberToIndex(idx, Y):
    return idx//Y, idx % Y
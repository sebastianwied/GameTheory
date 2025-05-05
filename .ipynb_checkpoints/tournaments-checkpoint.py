import numpy as np
import numpy.random as random
from memoryone import *
from roundtable import *
from evolution import *
from params import *
from time import perf_counter

def gridTournament(iterations, rounds, agents, neighborhoodSize, evolutionFunction, error=0):
    adjacencies = gridAdjacencyTable(agents, neighborhoodSize)
    Y, X = agents.shape
    # Setup snapshotting and scores
    scores = np.zeros(agents.shape)
    scoreSnaps = []
    agentSnaps = [agentArraySnapshot(agents)]
    agentRuleSnaps = [agentRuleSnapshot(agents)]
    # DO SNAPSHOTS
    for rnd in range(rounds):
        start = perf_counter()
        games = gridGenerateRound(adjacencies, agents)
        roundend = perf_counter()
        print('Generate round: ',roundend-start)
        scoresGained = np.zeros(agents.shape)
        for game in games:
            p1y, p1x = gridNumberToIndex(game[0], Y)
            p2y, p2x = gridNumberToIndex(game[1], Y)
            a1 = agents[p1y, p1x]
            a2 = agents[p2y, p2x]
            s1, s2 = playNIterations(a1, a2, iterations, error=error)
            scoresGained[p1y, p1x] += s1
            scoresGained[p2y, p2x] += s1
        # Evolution goes here
        agents = evolutionFunction(agents, adjacencies, scoresGained)
        scores += scoresGained
        scoreSnaps.append(scoresGained)
        agentSnaps.append(agentArraySnapshot(agents))
        agentRuleSnaps.append(agentRuleSnapshot(agents))
        end = perf_counter()
        print(end-start)
    scoreSnaps.append(np.copy(scores))
    e = perf_counter()
    return scores, scoreSnaps, agentSnaps, np.array(agentRuleSnaps), agents

def agentArraySnapshot(agents):
    agentSnap = np.zeros(agents.shape)
    for idr, row in enumerate(agents):
        for idc, ag in enumerate(row):
            agentSnap[idr,idc] = agentTypes[ag.name][1]
    return agentSnap

def agentRuleSnapshot(agents):
    agentRuleSnap = np.empty((agents.shape[0], agents.shape[1], 4))
    for idr, row in enumerate(agents):
        for idc, ag in enumerate(row):
            agentRuleSnap[idr,idc,:] = ag.rule
    return agentRuleSnap
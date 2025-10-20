import numpy as np
from numpy import random as rand
from math import ceil

# row index is my move, col index is their move
payoffMatrix = [[1, 5],
               [0, 4]] 

def torusTournament(agentGrid, iters, rounds):
    yLen, xLen = agentGrid.shape
    N = yLen*xLen
    # Choosing opponents
    def pickOpponents(N):
        angles = rand.random(N)*2*np.pi
        x = np.round(np.cos(angles))
        y = np.round(np.sin(angles))
        opponent = []
        for idy, row in enumerate(agentGrid):
            temp = []
            for idx, _ in enumerate(row):
                id = (idy*xLen)+idx
                xLoc = int((idx + x[id]) % xLen)
                yLoc = int((idy + y[id]) % yLen)
                temp.append((xLoc, yLoc))
            opponent.append(temp)
        # Return grid of tuples holding x,y position of each opponent.
        return opponent
    scoreSnaps = []
    ruleSnaps = []
    totalScore = np.zeros(agentGrid.shape)
    nonCumulativeScore = []
    snapEvery = ceil(rounds/100)
    paddedScore =np.empty((yLen+2, xLen+2))
    for round in range(rounds):
        # PLAY MATCHES
        ############################################################################
        matchups = pickOpponents(N)
        playedTracker = np.zeros(agentGrid.shape)
        scoreTracker = np.zeros(agentGrid.shape)
        for idy, row in enumerate(matchups):
            for idx, match in enumerate(row):
                # select players
                a1 = agentGrid[idy,idx]
                a2 = agentGrid[*match]
                # count game as played for both of them.
                playedTracker[idy,idx] += 1
                playedTracker[*match] += 1
                # Play N iterations
                seeds = rand.random(2*iters)
                for n in range(iters):
                    a1prev = a1.prev
                    a2prev = a2.prev
                    a1move = a1.playMove(a2prev, seeds[n], n)
                    a2move = a2.playMove(a1prev, seeds[n+iters], n)
                    scoreTracker[idy,idx] += payoffMatrix[a1move][a2move]
                    scoreTracker[*match] += payoffMatrix[a2move][a1move]
                a1.reset()
                a2.reset()
                
        # Make sure players don't get extra points for happening to get selected by more opponents
        scoreTracker /= playedTracker
        totalScore += scoreTracker
        ############################################################################
        # Handle Strategy Evolution
        # Evolution model:
        # Look at neighbors and pick neighbor with highest score from last round.
        # Move your strategy towards theirs by some percentage.
        # No mutation for now
        newGrid = agentGrid.copy()
        shiftPercentage = 0.2
        mutationRate = 0.01
        # Toroidal grid
        paddedScore[1:-1, 1:-1] = scoreTracker
        paddedScore[0, 1:-1] = scoreTracker[-1,:]
        paddedScore[-1, 1:-1] = scoreTracker[0,:]
        paddedScore[1:-1,0] = scoreTracker[:,-1]
        paddedScore[1:-1,-1] = scoreTracker[:,0]
        paddedScore[0,0] = scoreTracker[-1,-1]
        paddedScore[-1,-1] = scoreTracker[0,0]
        paddedScore[0,-1] = scoreTracker[-1,0]
        paddedScore[-1,0] = scoreTracker[0,-1]
        evolve = rand.random(N)
        chance = 0.1
        count = -1
        # find agents with max score around each agent
        for idy, row in enumerate(agentGrid):
            for idx, agent in enumerate(row):
                count += 1
                # Find max score index
                yMin, yMax = (idy), (idy+3)
                xMin, xMax = (idx), (idx+3)
                paddedCoords = np.argmax(paddedScore[yMin:yMax, xMin:xMax])
                unwrappedCoords = np.array([(paddedCoords // 3)-1+idy, (paddedCoords % 3)-1+idx])
                x, y = unwrappedCoords
                if y < 0:
                    y = yLen + y
                elif y >= yLen:
                    y -= 1
                if x < 0:
                    x = xLen + x
                elif x >= xLen:
                    x -= 1
                # Shift to high scoring neighbor
                ruleShift = np.array(agent.rule)*0
                if evolve[count] < chance:
                    ruleShift = (np.array(agentGrid[y,x].rule) - np.array(agent.rule)) * shiftPercentage
                # Mutate
                ruleShift2 = (rand.random(4**(agent.maxN)) * 2) - 1
                ruleShift2 *= mutationRate
                newRule = newGrid[idy, idx].rule + ruleShift + ruleShift2
                ruleShift = [max(0,min(1,i)) for i in newRule]
                newGrid[idy, idx].rule = ruleShift
                
        agentGrid = newGrid.copy()
        if (round == 1):
            scoreSnaps.append(np.copy(totalScore))
            ruleSnaps.append(np.copy(agentRuleSnapshot(agentGrid).copy()))
            nonCumulativeScore.append(scoreTracker.copy())
        if (round % snapEvery == 0) and (round//snapEvery) > 0:
            scoreSnaps.append(np.copy(totalScore))
            ruleSnaps.append(np.copy(agentRuleSnapshot(agentGrid)))
            nonCumulativeScore.append(scoreTracker.copy())
            print(round // snapEvery)
        
        
        ############################################################################
    return np.array(scoreSnaps), np.array(totalScore), np.array(ruleSnaps), np.array(nonCumulativeScore)

def torusTournamentNoEvolution(agentGrid, iters, rounds):
    yLen, xLen = agentGrid.shape
    N = yLen*xLen
    # Choosing opponents
    def pickOpponents(N):
        angles = rand.random(N)*2*np.pi
        x = np.round(np.cos(angles))
        y = np.round(np.sin(angles))
        opponent = []
        for idy, row in enumerate(agentGrid):
            temp = []
            for idx, _ in enumerate(row):
                id = (idy*xLen)+idx
                xLoc = int((idx + x[id]) % xLen)
                yLoc = int((idy + y[id]) % yLen)
                temp.append((xLoc, yLoc))
            opponent.append(temp)
        # Return grid of tuples holding x,y position of each opponent.
        return opponent
    scoreSnaps = []
    ruleSnaps = []
    nonCumulativeScore = []
    totalScore = np.zeros(agentGrid.shape)
    snapEvery = ceil(rounds/100)
    paddedScore =np.empty((yLen+2, xLen+2))
    for round in range(rounds):
        # PLAY MATCHES
        ############################################################################
        matchups = pickOpponents(N)
        playedTracker = np.zeros(agentGrid.shape)
        scoreTracker = np.zeros(agentGrid.shape)
        for idy, row in enumerate(matchups):
            for idx, match in enumerate(row):
                # select players
                a1 = agentGrid[idy,idx]
                a2 = agentGrid[*match]
                # count game as played for both of them.
                playedTracker[idy,idx] += 1
                playedTracker[*match] += 1
                # Play N iterations
                seeds = rand.random(2*iters)
                for n in range(iters):
                    a1prev = a1.prev
                    a2prev = a2.prev
                    a1move = a1.playMove(a2prev, seeds[n], n)
                    a2move = a2.playMove(a1prev, seeds[n+iters], n)
                    scoreTracker[idy,idx] += payoffMatrix[a1move][a2move]
                    scoreTracker[*match] += payoffMatrix[a2move][a1move]
                a1.reset()
                a2.reset()
                
        # Make sure players don't get extra points for happening to get selected by more opponents
        scoreTracker /= playedTracker
        totalScore += scoreTracker
        if (round == 1):
            scoreSnaps.append(np.copy(totalScore))
            ruleSnaps.append(np.copy(agentRuleSnapshot(agentGrid).copy()))
            nonCumulativeScore.append(scoreTracker.copy())
        if (round % snapEvery == 0) and (round//snapEvery) > 0:
            scoreSnaps.append(np.copy(totalScore))
            ruleSnaps.append(np.copy(agentRuleSnapshot(agentGrid)))
            nonCumulativeScore.append(scoreTracker.copy())
            print(round // snapEvery)
        
        
        ############################################################################
    return scoreSnaps, totalScore, ruleSnaps, nonCumulativeScore


def agentRuleSnapshot(agents):
    agentRuleSnap = np.empty((agents.shape[0], agents.shape[1], len(agents[0,0].rule)))
    for idr, row in enumerate(agents):
        for idc, ag in enumerate(row):
            agentRuleSnap[idr,idc,:] = ag.rule
    return agentRuleSnap
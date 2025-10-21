import numpy.random as random
# Constants:
COOP = 1
DEF = 0
# Interaction matrix entries
T = 5
R = 3 
P = 1 
S = 0

def distributePrize(me, them):
    if me == them == COOP:
        return R
    if (me == COOP) and (them == DEF):
        return S
    if (me == DEF) and (them == COOP):
        return T
    return P

def playOneIteration(agent1, agent2, roundNumber):
    '''
    Inputs: Two agent objects, iteration round number
    Output: Tuple of scores given to each player
    '''
    move1 = agent1.rule(agent2.past, roundNumber)
    move2 = agent2.rule(agent1.past, roundNumber)
    agent1.past.append(move1)
    agent2.past.append(move2)
    a1prize = distributePrize(move1, move2)
    a2prize = distributePrize(move2, move1)
    return a1prize, a2prize

def playNIterations(agent1, agent2, N):
    '''
    Inputs: Two agent objects, number of iterations
    Outputs: Tuple of scores accumulated for each player
    '''
    a1score = 0
    a2score = 0
    for n in range(N):
        a1p, a2p = playOneIteration(agent1, agent2, n)
        a1score += a1p
        a2score += a2p
    agent1.past = []
    agent2.past = []
    return a1score, a2score

class Agent:
    def __init__(self, short=True):
        self.past = []
        if short:
            self.short()
    
    def reset(self):
        self.score = 0
    
    def playMove(self, them, rn):
        return self.rule(them, rn)

    def addScore(self, result):
        self.score += result

    def getScore(self):
        return self.score

    def finishGame(self, score, outcome, enemy):
        # Outcome == 1 if won, 0 if lost
        self.againstAgentScore[enemy] = self.againstAgentScore.get(enemy, 0) + score
        self.againstAgentWins[enemy] = self.againstAgentWins.get(enemy, 0) + outcome
        self.reset()

    def __repr__(self):
        return self.name

class Cu(Agent):
    name = "Cooperate unconditionally"
    def rule(self, them, rn):
        return COOP
    def short(self):
        self.name = 'cu'

class Du(Agent):
    name = "Defect unconditionally"
    def rule(self, them, rn):
        return DEF
    def short(self):
        self.name = 'du'

class Rand(Agent):
    name = "Random"
    def rule(self, them, rn):
        return random.randint(0,2)
    def short(self):
        self.name = 'rand'

class Cp(Agent):
    name = "Cooperate probabilistically"
    def rule(self, them, rn):
        if random.rand() >= .75:
            return COOP
        return DEF
    def short(self):
        self.name = 'cp'

class TFT(Agent):
    name = "Tit for Tat"
    def rule(self, them, rn):
        if rn == 0: return COOP
        else: return them[-1]
    def short(self):
        self.name = 'tft'

class STFT(Agent):
    name = "Suspicious Tit for Tat"
    def rule(self, them, rn):
        if rn == 0: return DEF
        return them[-1]
    def short(self):
        self.name = 'stft'

class Grim(Agent):
    name = 'grim trigger'
    def rule(self, them, rn):
        if self.trigger and them == DEF:
            self.trigger = 0
        return self.trigger
    def short(self):
        self.name = 'grim trigger'
        self.trigger = 1

class Threes(Agent):
    def rule(self, them, rn):
        if rn%3 == 0: return COOP
        return DEF
    def short(self):
        self.name = 'threes'

def agentTypes():
    return {'du':Du, 'cu':Cu, 'rand':Rand, 'cp':Cp, 'tft':TFT, 'stft':STFT}
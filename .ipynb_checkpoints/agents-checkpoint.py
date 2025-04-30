import numpy as np
# Constants:
COOP = 1
DEF = 0
# Interaction matrix entries
payoff = (5,3,1,0)




class Agent:
    def __init__(self, short=True, score=0):
        self.past = []
        if short:
            self.short()
        self.score = score
    
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
        return np.random.randint(0,2)
    def short(self):
        self.name = 'rand'

class Cp(Agent):
    name = "Cooperate probabilistically"
    def rule(self, them, rn):
        if np.random.rand() >= .75:
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

MasterAgentTypes = {'du':(Du,0), 'cu':(Cu,1), 'rand':(Rand,2), 'cp':(Cp,3), 'tft':(TFT,4), 'stft':(STFT,5)}

MasterAgentIDs = {0:'du', 1:'cu', 2:'rand', 3:'cp', 4:'tft', 5:'stft'}

agentTypes = {'du':(Du,0), 'cu':(Cu,1), 'rand':(Rand,2), 'cp':(Cp,3), 'tft':(TFT,4), 'stft':(STFT,5)}

agentIDs = {0:'du', 1:'cu', 2:'rand', 3:'cp', 4:'tft', 5:'stft'}

def setAgents(selectFrom):
    newagentTypes = dict()
    newagentIDs = dict()
    for newID, agtype in enumerate(selectFrom):
        agName = agtype().name
        typeentry = MasterAgentTypes[agName]
        typeentry = (typeentry[0], newID)
        newagentTypes[agName] = typeentry
        newagentIDs[newID] = agName
    return newagentIDs, newagentTypes

def agentArraySnapshot(agents):
    agentSnap = np.zeros(agents.shape)
    for idr, row in enumerate(agents):
        for idc, ag in enumerate(row):
            agentSnap[idr,idc] = agentTypes[ag.name][1]
    return agentSnap
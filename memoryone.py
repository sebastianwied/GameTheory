import numpy as np
import numpy.random as random
COOP = 1
DEF = 0

class MemoryOne:
    # Rule: probability of cooperating after DD, DC, CD, CC (Me, Them)
    def __init__(self, score=0):
        self.score = score
        self.startup()

    def reset(self):
        self.score = 0

    def playMove(self, them, roundNum, error):
        prob = self.rule[(self.prev*2) + them]
        strat = (prob*(1-error))+(error*(1-prob))
        choice = COOP if random.rand() < strat else DEF
        return choice

    def addScore(self, result):
        self.score += result

    def getScore(self):
        return self.score

    def resetPast(self):
        self.prev = self.turnOne

    def __repr__(self): return self.name

class BLANK(MemoryOne):
    def startup(self):
        self.name = 'BLANK'
        self.rule = [0,0,0,0]
        self.turnOne = COOP if random.rand() < self.rule[0] else DEF
        self.prev = self.turnOne

class CU(MemoryOne): # Cooperate Unconditionally
    def startup(self):
        self.name = 'CU'
        self.rule = [1,1,1,1]
        self.turnOne = COOP
        self.prev = self.turnOne

class DU(MemoryOne): # Defect Unconditionally
    def startup(self):
        self.name = 'DU'
        self.rule = [0,0,0,0]
        self.turnOne = DEF
        self.prev = self.turnOne

class RAND(MemoryOne): # Random
    def startup(self):
        self.name = 'RAND'
        self.rule = [0.5,0.5,0.5,0.5]
        self.turnOne = COOP if random.rand() < self.rule[0] else DEF
        self.prev = self.turnOne

    def resetPast(self):
        self.prev = self.turnOne

class CP(MemoryOne): # Cooperate Probablistically
    def startup(self):
        self.name = 'CP'
        self.rule = [0.75,0.75,0.75,0.75]
        self.turnOne = COOP if random.rand() < self.rule[0] else DEF
        self.prev = self.turnOne

class TFT(MemoryOne): # Tit For Tat
    def startup(self):
        self.name = 'TFT'
        self.rule = [0,1,0,1]
        self.turnOne = COOP
        self.prev = self.turnOne
        self.score = 0

class STFT(MemoryOne): # Suspicious Tit for Tat
    def startup(self):
        self.name = 'STFT'
        self.rule = [0,1,0,1]
        self.turnOne = DEF
        self.prev = self.turnOne
        self.score = 0
    
class GRIM(MemoryOne): # Grim Trigger
    def startup(self):
        self.name = 'GRIM'
        self.rule = [0,0,0,1]
        self.turnOne = COOP
        self.prev = self.turnOne
        
    def playMove(self, them, roundNum, error):
        if them == DEF: self.rule = [0,0,0,0]
        prob = self.rule[(self.prev*2) + them]
        strat = (prob*(1-error))+(error*(1-prob))
        choice = COOP if random.rand() < strat else DEF
        return choice

    def resetPast(self):
        self.rule = [0,0,0,1]
        self.prev = COOP

MasterAgentTypes = {'BLANK':(BLANK,0), 'CU':(CU,1), 'RAND':(RAND,2), 'DU':(DU,3), 'TFT':(TFT,4), 'STFT':(STFT,5), 'GRIM':{GRIM,6}, 'CP':(CP,7)}

MasterAgentIDs = {0:'BLANK', 1:'CU', 2:'RAND', 3:'DU', 4:'TFT', 5:'STFT', 6:'GRIM', 7:'CP'}

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
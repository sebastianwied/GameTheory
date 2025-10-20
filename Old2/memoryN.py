import numpy as np
import numpy.random as random

COOP = 1
DEF = 0

class MemoryN:
    """
    MemoryN strategy is a 4^N long list of probabilities corresponding to
    the past n moves.
    N = 0: Doesnt look at past. Rule is 1 element.
    N = 1: Looks at prev move. 2 options for me, 2 for them. 4 choices
    N = 2: Looks at prev 2 moves. 4 options for me, 4 for them. 16 choices
    
    MemoryN agents have a predefined string of initial moves. Maybe something to change in future,
    but for now this removes weird edge cases with thinking the other agent defected when
    they havent done anything.
    
    All strategies in tournament have same N rule.
    """
    def __init__(self, N, startString, maxN):
        # N = number of moves to look into the past
        self.N = N
        self.maxN = maxN
        # Set up agent. Defined in the declaration of the agent.
        self.startup(startString)
    
    def addToPast(self, move):
        # Keeps whole past in the form of an integer. Data in the binary representation. 0=def,1=coop.
        # lsb is most recent move, msb is first move.
        self.prev = ((self.prev << 1) + move) & ((2**self.maxN)-1)
        #self.printPast()
        
    def playMove(self, theirPast, seed, roundNum):
        if roundNum < self.N:
            return self.startString & (2**roundNum)
        # Rulekey: N msb's is my past, N other bits are their past. 
        ruleKey = ((self.prev&(2**self.N - 1)) << self.N) + (theirPast&(2**self.N-1))
        prob = self.rule[ruleKey]
        if seed < prob:
            self.addToPast(COOP)
            return COOP
        self.addToPast(DEF)
        return DEF
    
    def reset(self):
        self.score = 0
        self.prev = self.startString
        
    def printPast(self):
        fSpec = f"0{self.N}b"
        print(f"past: binary: {self.prev:{fSpec}}, int: {self.prev}")

    def __repr__(self): return self.name
    
class BLANK(MemoryN):
    def setRule(self, rule):
        self.rule = rule
        
    def startup(self, startString):
        self.name = "BLANK"
        # Rule is 4^n long list of probabilities for each possible past scenario
        self.rule = np.zeros(4**self.maxN)
        self.startString = startString
        self.prev = self.startString

class TFT(MemoryN):
    def startup(self, startString):
        self.name = "TFT"
        self.N = 1
        self.rule = np.zeros(4**self.maxN)
        self.rule[1::2] = 1 # All odd index cooperate(lsb of ruleKey was cooperate)
        self.startString = COOP
        self.prev = COOP
        

class DU(MemoryN): #Defect unconditionally
    def startup(self, startString):
        self.name = "DU"
        self.N = 0
        self.rule = np.zeros(4**self.maxN)
        self.startString = DEF
        self.prev = DEF

class CU(MemoryN): # Cooperate unconditionally
    def startup(self, startString):
        self.name = "CU"
        self.N = 0
        self.rule = np.ones(4**self.maxN)
        self.startString = COOP
        self.prev = COOP

class Rand(MemoryN): # Random
    def startup(self, startString):
        self.name = "RAND"
        self.N = 0
        self.rule = np.ones(4**self.maxN) / 2
        self.startString = COOP if random.rand() < 0.5 else DEF
        self.prev = self.startString
    
    def __repr__(self): return f"{self.name} {self.startString}"

class _Template(MemoryN):
    def startup(self):
        pass
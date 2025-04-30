from params import *
import numpy.random as random

class MemoryOne:
    def __init__(self):
        self.prev = COOP
        self.score = 0
        # Rule: probability of cooperating after DD, DC, CD, CC (Me, Them)

    def reset(self):
        self.score = 0

    def playMove(self, them, roundNum, error):
        prob = self.rule[(self.prev*2) + them]
        strat = (prob*(1-error))+(error*(1-prob))
        choice = 1 if random.rand() < strat
        return choice

    def addScore(self, result):
        self.score += result

    def getScore(self):
        return self.score

    def __repr__(self): return self.name

def CU(MemoryOne):
    name = 'CU'
    rule = [1,1,1,1]

def DU(MemoryOne):
    name = 'DU'
    rule = [0,0,0,0]

def RAND(MemoryOne):
    name = 'RAND'
    rule = [0.5,0.5,0.5,0.5]

def CP(MemoryOne):
    name = 'CP'
    rule = [0.75,0.75,0.75,0.75]

def TFT(MemoryOne):
    name = 'TFT'
    rule = [0,1,0,1]

def GRIM(MemoryOne):
    name = 'TFT'
    rule = [0, 0, 0, 1]
    def playMove(self, them, roundNum, error):
        if them == DEF: self.rule = [0,0,0,0]
        prob = self.rule[(self.prev*2) + them]
        strat = (prob*(1-error))+(error*(1-prob))
        choice = 1 if random.rand() < strat
        return choice

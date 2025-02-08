import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# Constants:
COOP = 1
DEF = 0
T = 5
R = 3
P = 2
S = 1

def distributePrize(me, them):
    if (me and them) == COOP:
        return R
    if (me == COOP) and (them == DEF):
        return S
    if (me == DEF) and (them == COOP):
        return T
    return P

class Agent:
    def __init__(self):
        self.score = 0
        self.past = []
    
    def reset(self):
        self.score = 0
    
    def playMove(self, them, rn):
        return self.rule(them, rn)

    def getScore(self, result):
        self.score += result

class Arbiter:
    def __init__(self, agent1, agent2):
        self.a1 = agent1
        self.a2 = agent2
        self.a1wins = 0
        self.a2wins = 0
        self.ties = 0
        self.roundnum = 1

    def playGame(self):
        a1 = self.a1.playMove(self.a2.past, self.roundnum)
        a2 = self.a2.playMove(self.a1.past, self.roundnum)
        self.a1.getScore(distributePrize(a1, a2))
        self.a2.getScore(distributePrize(a2, a1))
        self.a1.past.append(a1)
        self.a2.past.append(a2)

    def playNTimes(self, N):
        self.a1.reset()
        self.a2.reset()
        for _ in range(N):
            self.playGame()
            self.roundnum += 1
        self.countWin()
    
    def showWins(self):
        print(f"A1 wins: {self.a1wins}, A2 wins: {self.a2wins}, Ties: {self.ties}")

    def countWin(self):
        a1 = self.a1.score
        a2 = self.a2.score
        #print(f"A1T: {a1T}, A2T: {a2T}, A1R: {a1R}, A2R: {a2R}, A1P: {a1P}, A2P: {a2P}, A1S: {a1S}, A2S: {a2S}")
        winner = 0
        #if a1T > a2T:
        #    winner = 1
        #elif a1T < a2T:
        #    winner = 2
        #elif a1R > a2R:
        #    winner = 1
        #elif a1R < a2R:
        #    winner = 2
        #elif a1P > a2P:
        #    winner = 1
        #elif a1P < a2P:
        #    winner = 2
        #elif a1S > a2S:
        #    winner = 1
        #elif a1S < a2S:
        #    winner = 2
        if a1 > a2: winner = 1
        elif a2 > a1: winner = 2
        if winner == 1: self.a1wins += 1; #print("A1 wins!")
        elif winner == 2: self.a2wins += 1;# print("A2 wins!")
        else: self.ties += 1; #print("Tie...")

class Cu(Agent):
    def rule(self, them, rn):
        #print(COOP)
        return COOP

class Du(Agent):
    def rule(self, them, rn):
        #print(DEF)
        return DEF

class Rand(Agent):
    def rule(self, them, rn):
        return random.randint(0,2)

class Cp(Agent):
    def rule(self, them, rn):
        if random.rand() >= .75:
            return COOP
        return DEF

class TFT(Agent):
    def rule(self, them, rn):
        if rn == 1: return COOP
        return them[-1]

class STFT(Agent):
    def rule(self, them, rn):
        if rn == 1: return DEF
        return them[-1]

class GrdTFT(Agent):
    def __init__(self):
        self.retal = 0
        self.apologize = False
        self.apologizeTurns = 2
        self.score = 0
        self.past = []
    
    def rule(self, them, rn):
        if rn == 1:  # Always cooperate in the first round
            return COOP

        # If apologizing, cooperate for `apologizeTurns` rounds
        if self.apologize:
            self.apologizeTurns -= 1
            if self.apologizeTurns == 0:  # End apology after 2 rounds
                self.apologize = False
            return COOP

        # If retaliating, defect
        if self.retal > 0:
            self.retal -= 1
            return DEF

        # Otherwise, mirror the opponent's last move
        if them[-1] == DEF:
            self.retal += 1  # Increase retaliation for each defection
            self.apologize = True  # Trigger apology after retaliation is done
            self.apologizeTurns = 2  # Reset apologyTurns
            return DEF
        
        return COOP  # Default to cooperating if no retaliation or apology


a1 = Du()
a2 = TFT()
arbiter = Arbiter(a1, a2)
for _ in range(10):
    arbiter.playNTimes(100)
arbiter.showWins()

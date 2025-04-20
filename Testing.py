import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import numpy.random as random
import math
from agents import *
from numba import njit
from time import perf_counter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim

class Arbiter:
    '''
    Manage running tournament. Base class. 
    Inherited, inheriting class defines how games are decided.
    Inheriting class also decides how the agent array is created. Agent array needs to be created at initialization
    '''
    def manageTournament(self, Nrounds):
        for _ in range(Nrounds):
            self.manageRound(self.generateRound())

    def manageRound(self, games):
        '''
        games: Array of 2-long arrays of matchups
        scores array: row is agent, column is opponent
        '''
        scores = np.zeros((len(self.agents),len(self.agents)))
        for game in games:
            a1, a2 = game[0], game[1]
            a1score, a2score = playNIterations(self.agents[a1],self.agents[a2], self.iters)
            scores[game[0], game[1]] += a1score
            scores[game[1], game[0]] += a2score
        self.score += scores

from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d

def perlinGrid(X, Y):
    agents = list(agentTypes().values())
    random.shuffle(agents)
    noise = generate_fractal_noise_2d((Y,X), (1,1))
    noise *= len(agents)
    noise += len(agents)
    noise /= 2
    noise = np.floor(noise).astype(int)
    #print(noise)
    agentgrid = [[] for _ in range(Y)]
    #print(noise)
    for idr, row in enumerate(noise):
        for item in row:
            agentgrid[idr].append(agents[item][0]())
    agentgrid = np.array(agentgrid)
    return agentgrid

def noiseGrid(X, Y):
    agents = list(agentTypes().values())
    print(agents)
    grid = np.floor(random.rand(Y,X)*(len(agents))).astype(int)
    agentgrid = [[] for _ in range(Y)]
    for idr, row in enumerate(grid):
        for item in row:
            agentgrid[idr].append(agents[item]())
    agentgrid = np.array(agentgrid)
    return agentgrid

class GridArbiter(Arbiter):
    '''
    For managing grid based tournaments
    '''
    def __init__(self, iterations, agents, error=None, neighborhoodSize=1):
        self.iters = iterations
        self.agents = agents # 2d grid of agents
        self.originalAgents = np.copy(agents)
        self.score = np.zeros(agents.shape)
        self.buildAdjMat(neighborhoodSize=neighborhoodSize)
        self.error = error

    def buildAdjMat(self, neighborhoodSize=1):
        '''
        Creates a grid of tuples - ([neighbor i indices], [probability of interacting with i])
        Based on a moore neighborhood
        '''
        start = perf_counter()
        Y, X = self.agents.shape
        adj = [[([],[]) for _ in range(X)] for _ in range(Y)]
        for y in range(Y):
            for x in range(X):
                # Agent (x,y). [xMin, xMax), [yMin, yMax)
                xMin, xMax = (max(x-neighborhoodSize, 0), min(x+neighborhoodSize, X-1))
                yMin, yMax = (max(y-neighborhoodSize, 0), min(y+neighborhoodSize, Y-1))
                size = (xMax+1-xMin)*(yMax+1-yMin) - 1
                weights = np.ones(size) / size
                neighbors = []
                for yn in range(yMin, yMax + 1):
                    for xn in range(xMin, xMax + 1):
                        if (xn == x) and (yn == y): continue
                        neighbors.append(yn*Y + xn)
                adj[y][x] = (neighbors, weights)
        self.adjacencies = adj
        end = perf_counter()
        print(f'BuildAdjMat took {end-start}')

    def generateRound(self):
        '''
        Generate single tournament round
        '''
        start = perf_counter()
        Y, X = len(self.agents), len(self.agents[0])
        N = X*Y
        players = set(range(N))
        games = []
        while len(players) > 1:
            choice = int(np.floor(random.rand()*N))
            while choice not in players:
                choice = int(np.floor(random.rand()*N))
            players.remove(choice)
            neighbors, weights = self.adjacencies[choice//Y][choice % Y]
            opponent = neighbors[random.choice(list(range(len(neighbors))), p=weights)]
            failed = True
            for _ in range(10):
                if opponent in players: 
                    failed = False
                    break
                opponent = neighbors[random.choice(list(range(len(neighbors))), p=weights)]
            if failed: 
                #print(f'Failed for {choice}')
                continue
            #print(choice, opponent, list(players))
            players.remove(opponent)
            games.append([choice, opponent])
        end = perf_counter()
        print(f'Generate round took {end-start}')
        return games

    def manageTournament(self, rounds, snapshots=False, evolution=False):
        if snapshots != False:
            scoreSnaps = []
            agentSnaps = [self.getAgentArraySnapshot()]
        Y, X= len(self.agents), len(self.agents[0])
        scores = np.zeros((Y,X))
        print(self.agents)
        for n in range(rounds):
            start = perf_counter()
            games = self.generateRound()
            for game in games:
                p1x, p1y = game[0] % X, game[0] // Y
                p2x, p2y = game[1] % X, game[1] // Y
                a1 = self.agents[p1y, p1x]
                a2 = self.agents[p2y, p2x]
                a1score, a2score = playNIterations(a1, a2, self.iters, self.error)
                scores[p1y, p1x] += a1score
                scores[p2y, p2x] += a2score
            newAgents = list(np.zeros(self.agents.shape))
            for idr, row in enumerate(self.agents):
                for idc, agent in enumerate(row):
                    neighbors = self.adjacencies[idr][idc][0]
                    neighborscores = [scores[neighbor // Y, neighbor % X] for neighbor in neighbors]
                    maxscore = max(neighborscores)
                    index = neighbors[neighborscores.index(maxscore)]
                    #print(agent.name, scores[idr, idc])
                    #print(neighbors, neighborscores)
                    #print(max(neighborscores))
                    if (random.rand() < .5) and scores[idr, idc] < maxscore:
                        self.agents[idr, idc] = type(self.agents[index // Y, index % X])(score=scores[idr, idc])
                        #print('switched')
                    #print()
            #print(self.agents)
            if snapshots != False: 
                scoreSnaps.append(np.copy(scores))
                agentSnaps.append(self.getAgentArraySnapshot())
            end = perf_counter()
            print(f'Round {n} took {end-start}')
        if snapshots != False:
            self.scoreSnaps = scoreSnaps
            self.agentSnaps = agentSnaps
        self.scores = scores
        self.rounds = rounds

    def getAgentArraySnapshot(self):
        agentSnap = np.zeros(self.agents.shape)
        agenttypes = agentTypes()
        for idr, row in enumerate(self.agents):
            for idc, ag in enumerate(row):
                agentSnap[idr,idc] = agenttypes[ag.name][1]
        return agentSnap

    def displayScores(self, text=True):
        fig, ax = plt.subplots(figsize=(64,64))
        plt.title(f'Rounds: {self.rounds}')
        plt.imshow(self.scores, cmap = 'YlOrRd')
        plt.colorbar()
        if text:
            for idr, row in enumerate(self.agents):
                for idx, agent in enumerate(row):
                    if idr % 5 != 0 or idx % 5 != 0: continue
                    ax.text(idx, idr, f'{agent.name}\n{self.scores[idr, idx]}', ha="center", va="center", color="k", rotation=-20)
            
    def displayScoreFrames(self, scoreFrames):
        fig, ax = plt.subplots(figsize=(8,8))
        im = plt.imshow(scoreFrames[0], cmap = 'YlOrRd', vmin = 0, vmax = np.max(scoreFrames[-1]))
        ani = anim.FuncAnimation(fig=fig, func=lambda frame: im.set_data(scoreFrames[frame]), frames=len(scoreFrames), interval=150)
        ani.save('noevolution.mp4', writer='ffmpeg', fps=5)
        plt.show()

    def displayAgentFrames(self, agentFrames):
        fig1, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,8))
        colorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        colormap = colors.LinearSegmentedColormap.from_list('custom', colorlist)
        orig = ax1.imshow(agentFrames[0], cmap=colormap, vmin=0, vmax=6)
        for idr, row in enumerate(agentFrames[0]):
                for idx, agent in enumerate(row):
                    ax1.text(idx, idr, f'{self.originalAgents[idr,idx].name}', ha="center", va="center", color="k", rotation=-20)
        im = ax2.imshow(agentFrames[0], cmap=colormap, vmin=0, vmax=6)
        ani = anim.FuncAnimation(fig=fig1, func=lambda frame: im.set_data(agentFrames[frame]), frames=len(agentFrames), interval=150)
        final = ax3.imshow(agentFrames[-1], cmap=colormap, vmin=0, vmax=6)
        for idr, row in enumerate(agentFrames[-1]):
                for idx, agent in enumerate(row):
                    ax3.text(idx, idr, f'{self.agents[idr,idx].name}', ha="center", va="center", color="k", rotation=-20)
        plt.colorbar(orig, ax=ax1)
        plt.colorbar(im, ax=ax2)
        plt.colorbar(final, ax=ax3)
        plt.show()
        
agents = perlinGrid(12,12)
grid = GridArbiter(10, agents, error=0.01, neighborhoodSize=1)
grid.manageTournament(10,snapshots=10)
grid.displayAgentFrames(grid.agentSnaps)
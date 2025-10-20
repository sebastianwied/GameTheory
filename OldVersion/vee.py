import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.random as random
import math
from agents import *

T, R, P, S = 7, 3, 1, 0


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
            #print(f'{a1}: {a1score}, {a2}: {a2score}')
            scores[game[0], game[1]] += a1score
            scores[game[1], game[0]] += a2score
        self.score += scores

    def displayScores(self):
        fig, ax = plt.subplots(figsize=(10,10))
        plt.imshow(self.score,cmap='plasma')
        plt.colorbar()
        plt.xlabel('agent')
        plt.ylabel('opponent')
        for x in range(self.score.shape[0]):
            for y in range(self.score.shape[1]):
                label = f'{self.agents[y]}({y}) vs {self.agents[x]}({x})\n{self.adjMat[y,x]:0.3f}'
                text = ax.text(x, y, label, ha="center", va="center", color="k", rotation=-45)

class RoundRobinArbiter(Arbiter):
    def __init__(self, iterations, agent_lst, rounds, payoffs, cull_number):
        self.iters = iterations
        self.agent_lst = agent_lst
        self.score_table = np.zeros((len(self.agent_lst),rounds))
        self.cull_number = cull_number
        self.rounds = rounds
        self.T = payoffs[0]
        self.R = payoffs[1]
        self.P = payoffs[2]
        self.S = payoffs[3]
    
    def generate_strategy_tracker(self):
        self.agent_strats = np.empty((self.rounds + 1,len(self.agent_lst)), dtype = 'object')
        for i in range(len(self.agent_lst)):
            self.agent_strats[0][i] = self.agent_lst[i].name
    
    def generate_score_tracker(self):
        self.score_matrix = np.zeros((self.rounds + 1, len(self.agent_lst)))
        self.score_matrix[0,:] = np.zeros((1,len(self.agent_lst)))
    
    def evolve_agents(self, round_number):
        idx = np.argpartition(self.score_matrix[round_number,:].ravel(), self.cull_number)
        lowScore_indices = np.array(np.unravel_index(idx, self.score_matrix[round_number,:].shape))[:, range(self.cull_number)].transpose().tolist()
        highScore_indices = np.argpartition(self.score_matrix[round_number,:], -1 * self.cull_number)[-1 * self.cull_number:]
        for i,j in zip(lowScore_indices, highScore_indices):
            #print(i)
            #print(j)
            self.agent_lst[i[0]] = self.agent_lst[j]
        for i in range(len(self.agent_lst)):
            self.agent_strats[round_number][i] = self.agent_lst[i].name
    
    def generate_tournament(self):
        self.generate_strategy_tracker()
        self.generate_score_tracker()
        for k in range(1,self.rounds+1):
            for i in range(len(self.agent_lst)):
                for j in range(1,len(self.agent_lst) - i):
                    iscore, jscore = playNIterations(self.agent_lst[i],self.agent_lst[i+j],self.iters)
                    self.score_matrix[k,i] += iscore
                    self.score_matrix[k,j+i] += jscore
            self.evolve_agents(k)
            
    def display_results(self):
        agent_types = []
        for i in self.agent_strats[0,:]:
            if i in agent_types:
                continue
            agent_types.append(i)
        agent_tracker = np.zeros((self.rounds+1, len(agent_types)))
        avg_score_tracker = np.zeros((self.rounds, len(agent_types)))
        
        for i in range(1,self.rounds+1):
            tracker_index = 0
            for j in agent_types:
                indices = [k for k, val in enumerate(self.agent_strats[i-1,:].tolist()) if val == j]
                agent_tracker[i-1,tracker_index] = len(indices)
                score_total = 0
                for k in indices:
                    score_total += self.score_matrix[i,k]
                if len(indices) == 0:
                    avg_score_tracker[i-1,tracker_index] = 0
                else:
                    avg_score_tracker[i-1,tracker_index] = score_total/len(indices)
                tracker_index+=1
        tracker_index = 0
        for i in agent_types:
            indices = [j for j, val in enumerate(self.agent_strats[-1,:].tolist()) if val == i]
            agent_tracker[-1,tracker_index] = len(indices)
            tracker_index += 1
        self.agent_type_lst = agent_types
        self.agent_counts = agent_tracker
        self.avg_scores = avg_score_tracker
        
        n_steps, n_bars = avg_score_tracker.shape
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax1, ax2 = axes
        ax1.set_title('Strategy Count')
        ax2.set_title('Strategy Average Score')
        for ax in axes:
            ax.set_xticks(np.arange(n_bars))
            ax.set_xticklabels(agent_types)
        ax1.set_ylim(0, sum(agent_tracker[0].tolist()))
        ax2.set_ylim(0, np.max(avg_score_tracker)*1.1)
        
        bars1 = ax1.bar(agent_types, agent_tracker[0,:], color='steelblue')
        bars2 = ax2.bar(agent_types, avg_score_tracker[0,:], color='orange')
        
        def update(frame):
            for bar, height in zip(bars1, agent_tracker[frame,:]):
                bar.set_height(height)
            for bar, height in zip(bars2, avg_score_tracker[frame,:]):
                bar.set_height(height)
            return bars1 + bars2
    
        ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=690, repeat=False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def agent_lst_maker(num_lst):
    agents_lst = []
    agents_lst = agents_lst + num_lst[0]*[Cu()]
    agents_lst = agents_lst + num_lst[1]*[Du()]
    agents_lst = agents_lst + num_lst[2]*[Rand()]
    agents_lst = agents_lst + num_lst[3]*[Cp()]
    agents_lst = agents_lst + num_lst[4]*[TFT()]
    agents_lst = agents_lst + num_lst[5]*[STFT()]
    agents_lst = agents_lst + num_lst[6]*[Grim()]
    return agents_lst

def input_agent_lst_maker():
    num_lst = []
    num_lst.append(int(input('\nHow many unconditionally cooperative players exist at the start of the simulation?\n')))
    num_lst.append(int(input('How many unconditionally defective players exist at the start of the simulation?\n')))
    num_lst.append(int(input('How many random players exist at the start of the simulation?\n')))
    num_lst.append(int(input('How many probablistically cooperative players exist at the start of the simulation?\n')))
    num_lst.append(int(input('How many tit for tat players exist at the start of the simulation?\n')))
    num_lst.append(int(input('How many suspicious tit for tat players exist at the start of the simulation?\n')))
    num_lst.append(int(input('How many grim trigger players exist at the start of the simulation?\n')))
    return agent_lst_maker(num_lst)

print('\n\n\n\n\n')
n_iters = int(input('How many matches will occur between each set of players per tournament round?\n'))
n_rounds = int(input('How many rounds will occur in the tournament?\n'))
cull_num = int(input('How many players are replaced after each round?\n'))
payoffs_inp =  input('Input numbers with no punctuation for the payoff matrix for the tournament.\n').split()
payoffs = [int(i) for i in payoffs_inp]
T, R, P, S = payoffs
agents_lst = input_agent_lst_maker()
rr_arbiter = RoundRobinArbiter(n_iters, agents_lst, n_rounds, payoffs, cull_num)
rr_arbiter.generate_tournament()
rr_arbiter.display_results()

#agents_lst = [Cu(),Du(),Cu(),Cu(),Cu(),Du(),Cp(),TFT(),TFT(),TFT(),TFT(),TFT(),TFT(),STFT(),STFT(),STFT(),Rand(),Rand(),Rand(),TFT()]
#agents_lst = agents_lst+agents_lst+agents_lst+agents_lst+agents_lst
#rr_arbiter = RoundRobinArbiter(100,agents_lst,10,[7,3,1,0],10)

#rr_arbiter.generate_tournament()
#rr_arbiter.display_results()
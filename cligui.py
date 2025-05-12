# Setting agents
import memoryone as m1
print(list(m1.MasterAgentIDs.items()))
agentIDstring = input('Input agent IDs as a list of numbers separated by spaces. If want to skip, type n: ')
if agentIDstring.strip() in ['n', 'N']:
    changingAgents = False
else:
    changingAgents = True
    agentIDs = agentIDstring.strip().split(' ')
    agentClasses = [m1.MasterAgentTypes[m1.MasterAgentIDs[int(ID)]][0] for ID in agentIDs]
    print(agentClasses)
import params
if changingAgents:
    params.agentIDs, params.agentTypes = m1.setAgents(agentClasses)

from roundtable import *
from agentgen import *
from tournaments import *
from displaying import *
from evolution import *
from time import perf_counter

gridShape = input('Input shape of agent grid as two numbers separated by a space: ')
shape = gridShape.strip().split(' ')
shape = (int(shape[0]), int(shape[1]))
gridRes = input('Input resolution of agent grid as two numbers separated by a space, divisor of previous: ')
res = gridRes.strip().split(' ')
res = (int(res[0]), int(res[1]))
genType = int(input('Perlin parameter grid (1) or perlin agent grid (2): ').strip())
genFunction = perlinParamGrid if genType == 1 else perlinGrid
agentgrid = genFunction((64,64), (4,4))
neighborhoodSize = int(input('Input neighborhood size - integer, less than shape: ').strip())
evolutionFunction = int(input('parameterEvolution (1) or basic evolution (2): ').strip())
evolutionFunction = gridParameterEvolution if evolutionFunction == 1 else gridBasicEvolution
adjacencies = gridAdjacencyTable(agentgrid, 1)
start = perf_counter()
scores, scoresnaps, agentsnaps, agentrulesnaps, agents = gridTournament(50, 50, agentgrid, 1, gridParameterEvolution)
end = perf_counter()
print(agentrulesnaps.shape)
if genType == 1:
    displayMemOneHist(agents, agentrulesnaps)
if genType == 2:
    displayMemoryOne(agents, agentsnaps, agentrulesnaps, scoresnaps)
print(end-start)
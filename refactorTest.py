import agents
selectFrom = [agents.TFT, agents.Cu, agents.Du, agents.Cp]
agents.agentIDs, agents.agentTypes = agents.setAgents(selectFrom)
agents.payoff = (6,3,1,0)

from roundtable import *
from agentgen import *
from tournaments import *
from displaying import *
from evolution import *
from time import perf_counter

agentgrid = perlinGrid((64,64),(4,4))
adjacencies = gridAdjacencyTable(agentgrid, 1)
start = perf_counter()
scores, scoresnaps, agentsnaps = gridTournament(100, 40, agentgrid, 1, gridBasicEvolution)
end = perf_counter()
displayFrames(agentsnaps, scoresnaps, 8, save=False)
print(end-start)
import memoryone as m1
selectFrom = [m1.TFT, m1.CU, m1.DU, m1.CP, m1.GRIM]
import params
params.agentIDs, params.agentTypes = m1.setAgents(selectFrom)

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
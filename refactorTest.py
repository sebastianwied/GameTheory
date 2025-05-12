import memoryone as m1
#selectFrom = [m1.TFT, m1.CU, m1.DU]#, m1.GRIM, m1.STFT]
import params
#params.agentIDs, params.agentTypes = m1.setAgents(selectFrom)

from roundtable import *
from agentgen import *
from tournaments import *
from displaying import *
from evolution import *
from time import perf_counter

agentgrid = perlinParamGrid((64,64), (4,4))
adjacencies = gridAdjacencyTable(agentgrid, 1)
start = perf_counter()
scores, scoresnaps, agentsnaps, agentrulesnaps, agents = gridTournament(50, 50, agentgrid, 1, gridBasicEvolution)
end = perf_counter()
print(agentrulesnaps.shape)
displayFrames(agentsnaps, scoresnaps, 8, False)
print(end-start)
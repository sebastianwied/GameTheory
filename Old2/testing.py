import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tournamentsNew import *
from agentGen import *
from display import *

# grid = np.array(
#     [[DU(1,0,1),DU(1,0,1),TFT(1,0,1),DU(1,0,1),DU(1,0,1)],
#     [DU(1,0,1),TFT(1,0,1),CU(1,0,1),TFT(1,0,1),DU(1,0,1)],
#     [TFT(1,0,1),CU(1,0,1),DU(1,0,1),CU(1,0,1),TFT(1,0,1)],
#     [DU(1,0,1),TFT(1,0,1),CU(1,0,1),TFT(1,0,1),DU(1,0,1)],
#     [DU(1,0,1),DU(1,0,1),TFT(1,0,1),DU(1,0,1),DU(1,0,1)]]
# )
grid = blankGrid(16,(4,4),1)
rounds = 500
iters = 40
scoreSnaps, totalScore, ruleSnaps, nonCumulativeScore = torusTournament(grid, iters, rounds)
print(ruleSnaps.shape)
print(scoreSnaps.shape)
#displayAsImage(scoreSnaps, totalScore, ruleSnaps)
contourPlots(nonCumulativeScore, totalScore, ruleSnaps, iters)
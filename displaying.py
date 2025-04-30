import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
import matplotlib.colors as colors
from params import *

def displayFrames(agentFrames, scoreFrames, labelfrequency, save=False):
    fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,8))
    # Agents
    colorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'yellow']
    colorlist = colorlist[:len(agentIDs.items())]
    indexcolorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'yellow', 'white']
    indexcolorlist = indexcolorlist[:len(agentIDs.items())]
    indexcolormap = colors.LinearSegmentedColormap.from_list('custom', indexcolorlist)
    colormap = colors.LinearSegmentedColormap.from_list('custom', colorlist)
    frame0 = agentFrames[0]
    bufferarr = np.ones((frame0.shape[0],1), dtype=int)*len(indexcolorlist)
    indexarr = np.linspace(0, len(indexcolorlist), frame0.shape[0], dtype=int).reshape(-1, 1)
    frame0 = np.hstack((frame0, bufferarr, indexarr))
    orig = ax1.imshow(frame0, cmap=indexcolormap, vmin=0, vmax=len(indexcolorlist))
    ax1.set_title("Original")
    prev = -1
    for i, val in enumerate(indexarr):
        if (val[0] > prev) and val[0] < len(colorlist) : 
            ax1.text(frame0.shape[1], i + (frame0.shape[0]//(2*len(colorlist))), f'{agentIDs[val[0]]}', ha="center", va="center", color="k", rotation=-20)
        prev = val[0]
    #for idr, row in enumerate(frame0):
    #        for idx, agent in enumerate(row):
    #            if not (idr % labelfrequency == 0 and idx % labelfrequency == 0): continue
    #            ax1.text(idx, idr, f'{agentIDs[frame0[idr,idx]]}', ha="center", va="center", color="k", rotation=-20)
    im = ax2.imshow(agentFrames[0], cmap=colormap, vmin=0, vmax=len(colorlist))
    ax2.set_title("Evolution")
    ani = anim.FuncAnimation(fig=fig1, func=lambda frame: im.set_data(agentFrames[frame]), frames=len(agentFrames), interval=150)
    final = ax3.imshow(agentFrames[-1], cmap=colormap, vmin=0, vmax=len(colorlist))
    ax3.set_title("Final")
    print(agentFrames[-1])
    for idr, row in enumerate(agentFrames[-1]):
            for idx, agent in enumerate(row):
                if not (idr % labelfrequency == 0 and idx % labelfrequency == 0): continue
                ax3.text(idx, idr, f'{agentIDs[agentFrames[-1][idr,idx]]}', ha="center", va="center", color="k", rotation=-20)
    plt.colorbar(orig, ax=ax1)
    plt.colorbar(im, ax=ax2)
    plt.colorbar(final, ax=ax3)
    # Scores
    score1 = ax4.imshow(scoreFrames[0], cmap='YlOrRd')
    ax1.set_title("Score after round 1")
    for idr, row in enumerate(agentFrames[0]):
            for idx, agent in enumerate(row):
                if not (idr % labelfrequency == 0 and idx % labelfrequency == 0): continue
                ax4.text(idx, idr, f'{agentIDs[agentFrames[0][idr,idx]]}', ha="center", va="center", color="k", rotation=-20)
    scoreim = ax5.imshow(scoreFrames[0], cmap='YlOrRd', vmin=0, vmax=np.max(scoreFrames[-1]))
    ax2.set_title("Score Evolution")
    aniscore = anim.FuncAnimation(fig=fig1, func=lambda frame: scoreim.set_data(scoreFrames[frame]), frames=len(scoreFrames), interval=150)
    finalscore = ax6.imshow(scoreFrames[-1], cmap='YlOrRd')
    ax3.set_title("Final")
    for idr, row in enumerate(scoreFrames[-1]):
            for idx, agent in enumerate(row):
                if not (idr % labelfrequency == 0 and idx % labelfrequency == 0): continue
                ax6.text(idx, idr, f'{agentIDs[agentFrames[-1][idr,idx]]}', ha="center", va="center", color="k", rotation=-20)
    plt.colorbar(score1, ax=ax4)
    plt.colorbar(scoreim, ax=ax5)
    plt.colorbar(finalscore, ax=ax6)
    if save:
        ani.save('evolution.mp4', writer='ffmpeg', fps=30)
    plt.show()
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

def displayMemoryOne(agents, agentFrames, agentRuleFrames, scoreFrames):
    framerate = 300
    fig1, ((ax1, ax2, ax3, ax7), (ax4, ax5, ax6, ax8)) = plt.subplots(2,4, figsize=(16,8))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax7.axis('off')
    ax8.axis('off')
    strategies = np.empty((agents.shape[0],agents.shape[0],4))
    for idr, row in enumerate(agents):
        for idc, agent in enumerate(row):
            strategies[idr][idc][:] = agent.rule
    DDframes = agentRuleFrames[:,:,:,0]
    DCframes = agentRuleFrames[:,:,:,1]
    CDframes = agentRuleFrames[:,:,:,2]
    CCframes = agentRuleFrames[:,:,:,3]
    imDD = ax2.imshow(DDframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('DD(Me, Them)')
    anidd = anim.FuncAnimation(fig=fig1, func=lambda frame: imDD.set_data(DDframes[frame]), frames=len(DDframes), interval=framerate)
    imDC = ax3.imshow(DCframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax3.set_title('DC(Me, Them)')
    anidc = anim.FuncAnimation(fig=fig1, func=lambda frame: imDC.set_data(DCframes[frame]), frames=len(DCframes), interval=framerate)
    imCD = ax5.imshow(CDframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax5.set_title('CD(Me, Them)')
    anicd = anim.FuncAnimation(fig=fig1, func=lambda frame: imCD.set_data(CDframes[frame]), frames=len(CDframes), interval=framerate)
    imCC = ax6.imshow(CCframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax6.set_title('CC(Me, Them)')
    anicc = anim.FuncAnimation(fig=fig1, func=lambda frame: imCC.set_data(CCframes[frame]), frames=len(CCframes), interval=framerate)
    plt.colorbar(imDD, ax=ax2)
    plt.colorbar(imDC, ax=ax3)
    plt.colorbar(imCD, ax=ax5)
    plt.colorbar(imCC, ax=ax6)

    colorlist = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'yellow']
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
    agim = ax4.imshow(agentFrames[-1], cmap=indexcolormap, vmin=0, vmax=len(indexcolorlist))
    aniag = anim.FuncAnimation(fig=fig1, func=lambda frame: agim.set_data(agentFrames[frame]), frames=len(agentFrames), interval=framerate)
    
    scoreim = ax7.imshow(scoreFrames[0], cmap='YlOrRd')
    ax7.set_title("Score Evolution")
    aniscore = anim.FuncAnimation(fig=fig1, func=lambda frame: scoreim.set_data(scoreFrames[frame]), frames=len(scoreFrames), interval=150)
    finalscore = ax8.imshow(scoreFrames[-1], cmap='YlOrRd')
    ax8.set_title("Final")
    
    plt.show()

def displayMemOneHist(agents, agentRuleFrames):
    framerate = 300
    fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(16,8))
    ax2.axis('off')
    ax3.axis('off')
    ax6.axis('off')
    ax7.axis('off')
    DDframes = agentRuleFrames[:,:,:,0]
    DCframes = agentRuleFrames[:,:,:,1]
    CDframes = agentRuleFrames[:,:,:,2]
    CCframes = agentRuleFrames[:,:,:,3]

    ddHist = ax1.hist(DDframes[0].ravel())
    ax1.set_xlim(0,1)
    def ddUpdate(frame):
        ax1.cla()
        ax1.hist(DDframes[frame].ravel())
    ddHistAni = anim.FuncAnimation(fig=fig1, func=ddUpdate, frames=len(DDframes), interval=framerate)

    imDD = ax2.imshow(DDframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('DD(Me, Them)')
    anidd = anim.FuncAnimation(fig=fig1, func=lambda frame: imDD.set_data(DDframes[frame]), frames=len(DDframes), interval=framerate)
    
    imDC = ax3.imshow(DCframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax3.set_title('DC(Me, Them)')
    anidc = anim.FuncAnimation(fig=fig1, func=lambda frame: imDC.set_data(DCframes[frame]), frames=len(DCframes), interval=framerate)
    
    dcHist = ax4.hist(DCframes[0].ravel())
    ax4.set_xlim(0,1)
    def dcUpdate(frame):
        ax4.cla()
        ax4.hist(DCframes[frame].ravel())
    dcHistAni = anim.FuncAnimation(fig=fig1, func=dcUpdate, frames=len(DCframes), interval=framerate)
    
    imCD = ax6.imshow(CDframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax6.set_title('CD(Me, Them)')
    anicd = anim.FuncAnimation(fig=fig1, func=lambda frame: imCD.set_data(CDframes[frame]), frames=len(CDframes), interval=framerate)
    
    cdHist = ax5.hist(CDframes[0].ravel())
    ax5.set_xlim(0,1)
    def cdUpdate(frame):
        ax5.cla()
        ax5.hist(CDframes[frame].ravel())
    cdHistAni = anim.FuncAnimation(fig=fig1, func=cdUpdate, frames=len(CDframes), interval=framerate)
    
    imCC = ax7.imshow(CCframes[0], cmap='YlOrRd', vmin=0, vmax=1)
    ax7.set_title('CC(Me, Them)')
    anicc = anim.FuncAnimation(fig=fig1, func=lambda frame: imCC.set_data(CCframes[frame]), frames=len(CCframes), interval=framerate)
    
    ccHist = ax8.hist(CCframes[0].ravel())
    ax8.set_xlim(0,1)
    def ccUpdate(frame):
        ax8.cla()
        ax8.hist(CCframes[frame].ravel())
    ccHistAni = anim.FuncAnimation(fig=fig1, func=ccUpdate, frames=len(CCframes), interval=framerate)
    
    plt.colorbar(imDD, ax=ax2)
    plt.colorbar(imDC, ax=ax3)
    plt.colorbar(imCD, ax=ax6)
    plt.colorbar(imCC, ax=ax7)
    plt.show()
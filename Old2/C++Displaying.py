import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

def load_csv(filename):
    """Load a CSV file and return a 2D numpy array."""
    data = np.loadtxt(filename, delimiter=',')
    return data

def displayAsImage(scoreSnaps, totalScore, ruleSnaps):
    fig1, axes = plt.subplots(3, 4, figsize=(16, 8))
    mspf = 60  # ms per frame

    # Unpack axes for clarity
    (ax1, ax2, fax1, fax2), (ax3, ax4, fax3, fax4), (ax5, ax6, fax5, fax6) = axes

    im = ax1.imshow(scoreSnaps[0], cmap="viridis", vmin=0, vmax=np.nanmax(totalScore))
    ax1.set_title("Score Evolution")
    plt.colorbar(im, ax=ax1)
    anim.FuncAnimation(fig1, func=lambda frame: im.set_data(scoreSnaps[frame]),
                       frames=len(scoreSnaps), interval=mspf)

    ax2.imshow(totalScore, cmap="viridis", vmin=0, vmax=np.nanmax(totalScore))
    ax2.set_title("Final Total Score")

    # Unpack rule components
    ruleR1s, ruleR2s, ruleR3s, ruleR4s = [ruleSnaps[..., i] for i in range(4)]

    # Create animations for rule evolution
    for (ax, data, title) in zip([ax3, ax4, ax5, ax6],
                                 [ruleR1s, ruleR2s, ruleR3s, ruleR4s],
                                 ["R1 evolution", "R2 evolution", "R3 evolution", "R4 evolution"]):
        im_rule = ax.imshow(data[0], cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im_rule, ax=ax)
        ax.set_title(title)
        anim.FuncAnimation(fig1, func=lambda frame, im=im_rule, d=data: im.set_data(d[frame]),
                           frames=len(scoreSnaps), interval=mspf)

    # Final snapshots
    for (fax, data, title) in zip([fax3, fax4, fax5, fax6],
                                  [ruleR1s, ruleR2s, ruleR3s, ruleR4s],
                                  ["R1 final", "R2 final", "R3 final", "R4 final"]):
        fax.imshow(data[-1], cmap="viridis", vmin=0, vmax=1)
        fax.set_title(title)

    # Hide unused axes
    for ax in [fax1, fax2, ax1, ax2, ax3, ax4, ax5, ax6, fax3, fax4, fax5, fax6]:
        ax.axis("off")

    plt.show()

def flattenDataToHist(data, max_val=None):
    histogramX = list(range(len(data)))
    histogramZ = []
    maxScore = np.nanmax(data[-1]) if max_val is None else max_val
    bins = np.linspace(0, maxScore, 200)
    for snap in data:
        hist, _ = np.histogram(snap, bins)
        histogramZ.append(hist)
    histogramY = [(bins[n] + bins[n+1]) / 2 for n in range(len(bins) - 1)]
    x, y = np.meshgrid(histogramY, histogramX)
    z = np.array(histogramZ)
    return x, y, z

def contourPlots(scoreSnaps, totalScore, ruleSnaps, iters):
    x, y, z = flattenDataToHist(scoreSnaps, iters * 5)
    z_masked = np.ma.masked_where(z == 0, z)
    fig = plt.figure(figsize=(12, 8))
    nrows, ncols = 2, 3
    score_ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
    score_ax.plot_surface(x, y, z_masked, cmap="gist_gray", lw=0.5, rstride=1, cstride=1)
    score_ax.set_xlabel("Round")
    score_ax.set_ylabel("Score bin")
    score_ax.set_title("Score Evolution")
    score_ax.view_init(elev=90, azim=0)

    labels = ["Rule 1", "Rule 2", "Rule 3", "Rule 4"]
    for i, label in enumerate(labels):
        x, y, z = flattenDataToHist(ruleSnaps[..., i], max_val=1)
        z_masked = np.ma.masked_where(z == 0, z)
        ax = fig.add_subplot(nrows, ncols, 2 + i, projection="3d")
        ax.plot_surface(x, y, z_masked, cmap="gist_gray", lw=0.5, rstride=1, cstride=1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Rule Value")
        ax.set_title(label)
        ax.view_init(elev=90, azim=0)
    plt.show()

if __name__ == "__main__":
    # Example: load from CSV files exported by your C++ simulation
    scoreSnaps = load_csv("scoreSnaps.csv")  # shape: (frames, grid_y, grid_x)
    totalScore = load_csv("totalScore.csv")      # shape: (grid_y, grid_x)
    ruleSnaps  = load_csv("ruleSnaps.csv")   # shape: (frames, grid_y, grid_x*4)
    iters = 200
    print(totalScore.shape)
    print(len(scoreSnaps))

    # Reshape ruleSnaps if stored flattened
    yLen, xLen = totalScore.shape
    ruleSnaps = ruleSnaps.reshape(iters, yLen, xLen, 4)
    scoreSnaps = scoreSnaps.reshape()

    displayAsImage(scoreSnaps, totalScore, ruleSnaps)
    contourPlots(scoreSnaps, totalScore, ruleSnaps, iters)

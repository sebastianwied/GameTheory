import sys
import csv
import subprocess
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from numpy.random import rand

# ── Parameters ────────────────────────────────────────────────────────────────
N         = 50
ROUNDS    = 1000000
ITERS     = 200
SEED      = 521#int(rand()*1000)
SNAPS     = 1000
EV_RATE   = 0.5
EV_CHANCE = 0.2
MUT_RATE  = 0.005

GRAPH_FILE  = "graph.csv"
RULES_FILE  = "rules.csv"
OUTPUT_FILE = "snapshots.bin"

# ── Generate network ──────────────────────────────────────────────────────────
#G = nx.watts_strogatz_graph(N, 4, 0.3, seed=SEED)
G = nx.connected_caveman_graph(int(N/10), 10)
G = nx.convert_node_labels_to_integers(G)

with open(GRAPH_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    for node in sorted(G.nodes):
        writer.writerow(sorted(G.neighbors(node)))

print(f"Wrote {GRAPH_FILE}  ({N} nodes, {G.number_of_edges()} edges)")

# ── Generate random rules ─────────────────────────────────────────────────────
# Each agent has 4 rule values in [0, 1]:
#   rules[0] = P(coop | I defected,   they defected)   DD
#   rules[1] = P(coop | I cooperated, they defected)   CD
#   rules[2] = P(coop | I defected,   they cooperated) DC
#   rules[3] = P(coop | I cooperated, they cooperated) CC
rng   = np.random.default_rng(SEED)
rules = rng.random((N, 4))
rules[:,0] = np.zeros(N)
rules[:,1] = np.ones(N)
rules[:,2] = np.zeros(N)
rules[:,3] = np.ones(N)

with open(RULES_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    for row in rules:
        writer.writerow(f"{x:.6f}" for x in row)

print(f"Wrote {RULES_FILE}")

# ── Run simulation ────────────────────────────────────────────────────────────
exe = "./sim" if sys.platform != "win32" else "sim.exe"
cmd = [exe, GRAPH_FILE, RULES_FILE, OUTPUT_FILE, str(ROUNDS), str(ITERS), str(SEED), str(SNAPS), str(EV_CHANCE), str(EV_RATE), str(MUT_RATE)]
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

# ── Read snapshots ────────────────────────────────────────────────────────────
# Each snapshot row: [round, cooperation_count, score_0..score_N-1, rule_00..rule_N-1_3]
# All stored as float64.
ROW = 1 + 1 + N + 4 * N
data        = np.fromfile(OUTPUT_FILE, dtype=np.float64).reshape(-1, ROW)
snap_rounds = data[:, 0].astype(int)
snap_coops = data[:, 1].astype(int)
snap_scores = data[:, 2 : N + 2]               # (S, N)
snap_rules  = data[:, N + 2 :].reshape(-1, N, 4) # (S, N, 4)

snap_rules_means = np.zeros((snap_rules.shape[0], 4))
print(snap_rules.shape)
print(snap_rules_means.shape)
for i in range(snap_rules.shape[0]):
    for j in range(4):
        snap_rules_means[i, j] = np.mean(snap_rules[i, :, j])
rounds_range = list(range(snap_rules.shape[0]))

num_snaps = len(snap_rounds)
print(f"Read {num_snaps} snapshots")

# ── Fixed network layout ──────────────────────────────────────────────────────
pos    = nx.spring_layout(G, seed=SEED)
xs     = np.array([pos[i][0] for i in range(N)])
ys     = np.array([pos[i][1] for i in range(N)])
edges  = list(G.edges())
edge_x = []
edge_y = []
for u, v in edges:
    edge_x += [pos[u][0], pos[v][0], None]
    edge_y += [pos[u][1], pos[v][1], None]

# ── Build figure ──────────────────────────────────────────────────────────────
RULE_LABELS = ["P(coop|DD)", "P(coop|CD)", "P(coop|DC)", "P(coop|CC)"]
ALL_LABELS  = RULE_LABELS + ["Rule Means", "Cumulative Score"]

#fig, axes = plt.subplots(2, 5, figsize=(22, 4.5))
fig = plt.figure(figsize=(22,8))
gs = GridSpec(3,4, figure=fig)
fig.patch.set_facecolor("#1a1a2e")

score_vmin = snap_scores.min()
score_vmax = snap_scores.max()
vmins = [0, 0, 0, 0, score_vmin]
vmaxs = [1, 1, 1, 1, score_vmax]
cmaps = ["RdYlGn", "RdYlGn", "RdYlGn", "RdYlGn", "plasma"]

scatters = [] 
for col in range(4):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor("#1a1a2e")
    ax.plot(edge_x, edge_y, color="#555577", linewidth=0.6, alpha=0.5)
    sc = ax.scatter(xs, ys, c=np.zeros(N), cmap=cmaps[col],
                    vmin=vmins[col], vmax=vmaxs[col],
                    s=120, zorder=3, edgecolors="#cccccc", linewidths=0.3)
    ax.set_title(ALL_LABELS[col], color="white", fontsize=10, pad=6)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    scatters.append(sc)

ax = fig.add_subplot(gs[1, 0:4])
ax.set_facecolor("#1a1a2e")
colors = ["red", "blue", "green", "orange"]
for i in range(4):
    ax.plot(rounds_range, snap_rules_means[:, i], color=colors[i], label=RULE_LABELS[i])
ax.legend()

# ax = fig.add_subplot(gs[1, 0:4])
# ax.set_facecolor("#1a1a2e")
# colors = ["red", "blue", "green", "orange"]
# for i in range(4):
#     ax.plot(rounds_range, snap_rules_means[:, i], color=colors[i], label=RULE_LABELS[i])
# ax.legend()



ax = fig.add_subplot(gs[0, 3])
ax.set_facecolor("#1a1a2e")
ax.plot(edge_x, edge_y, color="#555577", linewidth=0.6, alpha=0.5)
sc = ax.scatter(xs, ys, c=np.zeros(N), cmap=cmaps[4],
                vmin=vmins[4], vmax=vmaxs[4],
                s=120, zorder=3, edgecolors="#cccccc", linewidths=0.3)
ax.set_title(ALL_LABELS[4], color="white", fontsize=10, pad=2)
ax.set_aspect("equal")
ax.axis("off")
plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
scatters.append(sc)

title = fig.suptitle(f"Seed: {SEED}. Round 0", color="white", fontsize=13, y=0.95)
plt.tight_layout()

# ── Animation ─────────────────────────────────────────────────────────────────
def update(frame):
    for k in range(4):
        scatters[k].set_array(snap_rules[frame, :, k])
    scatters[4].set_array(snap_scores[frame])
    title.set_text(f"Seed: {SEED}. Round {snap_rounds[frame]}")
    return scatters + [title]

ani = animation.FuncAnimation(
    fig, update, frames=num_snaps, interval=10, blit=False
)

# Save gif (requires Pillow: pip install pillow)
plt.show()

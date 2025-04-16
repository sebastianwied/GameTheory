# Docs

## File and program structure
### agents.py
This file holds code universal to agents and interactions between them. Add new strategies here.
Besides agents, it contains the code to:
- Define and distribute rewards according to the payoff matrix
- Play n iterations between agents. This is the backbone of the game.
Strategy structure - When creating new strategies, you need to include the following precisely:
- name string
- rule definition, taking only opponent move history(1d array of 0 or 1) and round number.
- function to return short version of name

### DilemmaSim.ipynb
Most current version. Core class is *Arbiter*, every model of tournament inherits from this. 
Arbiter class:
- manageTournament:
  - Plays N rounds. Parameter N is integer
- manageRound:
  - Manages games between agents. Games is a list of 2 elements lists, each of which contains 2 agent indexes.
  - Updates scores array.
Other tournament types inherit from Arbiter.

Network Arbiter:
- Takes a 1d list of agents.
- Score table is a 2d table. $C_{ij}$ is the score of agent i against agent j.
- Methods:
    - constructTable: Constructs 2d adjacency matrix, where each column i is the probability of agent i interacting with every other agent
    - generateRound: Randomly selects agents and matches them against an opponent based on the adjacency table. Returns list of 2 element lists. Matches input to manageRound in arbiter class

Grid Arbiter:
- Takes a 2d matrix of agents of size equal to grid size.
- Probability to interact with other agents is uniform across moore neighborhood.
- Methods:
  - buildAdjMat: Builds grid of grids. Each element in the overall grid corresponds to that point in the agent grid. The contained grid is that agents adjacency matrix.
  - generateRound: Creates round based on the above adjacency table
  - manageTournament: manages the 2d grid game selection

### PrisonersDilemma.ipynb
This is the old version of the simulator. Only has round robin and 

### Roadmap
#### Features we'd like to add:
- Evolution
- 
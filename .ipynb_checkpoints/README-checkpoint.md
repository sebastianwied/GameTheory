# How to run
Go to refactorTest.py and change the lines of code in there to do what you want. First, generate agents. Second, generate adjacency tables. Third, run the tournament with the desired tournament framework. Supply the desired evolution model to the tournament function. At the end, plot the results using functions from displaying.py.

# Program Structure
## memoryone.py
This file is the preferred agent implementation. Implements every agent as a memoryone agent. Memory-n agents coming later.

Every strategy is represented by a *rule* - a 4 element array containing the probability to cooperate after certain interactions denoted by a tuple of (My move, Their move). The rule array follows the structure: [DD, DC, CD, CC]. This order is due to binary encoding of the interaction tuple where D=0 and C=1. This makes choosing a move easier, but may be different than other literature specifying memory-one strategies. Apologies.

Strategies making permanent changes is done by redefining the playMove function to implement the desired functionality. See GRIM for an example.

Add new strategies by following the format of the existing strategies. This only supports memory-1 strategies.

The first move an agent makes is given by the parameter turnOne

agents.py is the old implementation. Stay away please.

## params.py
Contains all parameters necessary for changing during the simulation setup.

## roundtable.py
Manages all adjacency table and round generation functions, including individual matchup funcitonality.

Add tournament substructure here. A tournament substructure consists of a:
- generateRound function which manages generating a round of the tournament based on a supplied adjacency table. Implementation of this is up to you.
- adjacencyTable function which manages generating the adjacency table. Also up to you
Feel free to add helper functions

## tournaments.py
Manages the overall tournament functionality.

Add the overall tournament structure here. Imports substructure from roundtable.py. See gridTournament for implementation details.

gridTournament:
- Takes N iterations per game, N games, an agent 2d grid, integer moore neighborhood size, and the function for handling evolution. Import this from evolution.py

## evolution.py
Put evolution functionality here.

## displaying.py
Contains all visualization functionality.

displayFrames:
- Built for grid tournaments, animates both agent evolution and score progression of the whole grid using lists of score snapshots and agent snapshots. Tiles the grid with labels based on the integer labelfrequency. Saves to an output file if save != False.

## agentgen.py
Contains methods to generate agent matrices. Add more functions here if you want.

perlinGrid:
- generates agent grid based on perlin noise.

noiseGrid:
- generates grid based on purely random noise.
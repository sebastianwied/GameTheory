# Required Python Packages
- numpy (I am on 2.2.6)
- networkx (I am on 3.6.1)
- matplotlib (I am on 3.10.3)

# How to run
Not sure if it works on windows. On macos, make sure gcc is installed. I am on Apple Clang Version 17.0.0 (clang-1700.6.3.2).

In command line, create virtual environment in repo directory and install above packages using pip.

Activate command line, then run ```make```. This will compile the sim.cpp file. Next, run ```python simulate.py``` and let it run. You don't have to run make for each simulation run, only when you change the cpp file.

# Changing parameters
- Main simulation parameters are in the top of the python file. Just change the number. Keep the seed the same to reproduce the exact same rng. 
- Initial world state defined by an array of 4 arrays: the rules array. To change the initial state, just generate a different numpy array for that rule.

# Notes
Some things are not working fully yet, update if you see new commits.
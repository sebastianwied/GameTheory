# Spatial Iterated Prisoners Dilemma
## Problem and simulation intro
https://docs.google.com/presentation/d/1tgJJWcj52KPbl12UBf70QUW3kY9BSJxe5ExPTy3HqtY/edit?usp=sharing

## Setup
Required packages:
- Matplotlib (current version: 3.10.3)
- Numpy (current version: 2.2.6)
- Pandas (current version: 2.3.3)

Beyond python setup, the ```sim.cpp``` file needs to be compiled. On mac, the command I'm using to compile is:

```g++ -O3 -std=c++17 sim.cpp -o sim -pthread```

Replace sim.cpp with simEvolvingMR.cpp or simGlobalMR.cpp. EvolvingMR has mutationrate evolution turned off, model wasn't working.

Not sure on windows. Ask chatgpt or something!

## Running
The main functionality is provided by experiments.py. This has functionality for defining simulation runs, running them, and displaying the outputs, all while keeping data outputs nice and organized.

The core of the experiment functionality is the **ExperimentBuilder** class. Instantiation only takes one argument: the local path to the simulation binary. Under my current file structure, the code for creating an ExperimentBuilder is:

```ExperimentBuilder("./sim")```

The ExperimentBuilder object will keep an internal list of all the experiments created by the object. 
The functionality for running the experiments runs off this internal list, though the experiment creation functions return the experiments as objects if you would like to do any special further handling with them.

### Explanation of the overall functionality
The way you interact with this simulation is through **experiments**. These are objects which define all of the input parameters for a simulation. When the experiment object is created, it will create along with it a folder within the ```./Data``` directory. Once ran, the experiment will put all the output data into it's folder in here. It will also have a csv file containing the parameters for the simulation. Default naming for these is ```exp[N]```, where ```[N]``` is a unique number which is one higher than the previous highest one existing in the Data directory. Each repeat for the experiment is given a subfolder called ```repeat[N]``` where ```[N]``` is the repeat number.

The experimentBuilder class provides functionality for easily constructing and handling these experiments.

Example start to end setting up to displaying:
```
builder = ExperimentBuilder("./simTest")
exps = builder.payoffMatrixRange(3,4,5,(1,1))
builder.runAll()
builder.experimentList()
fromTracker()
```

### Methods
#### Experiment creation
- Most simply, there is **fromParamDict()**. This function takes in as its only parameter a dictionary containing every parameter you would like to set for an experiment. Any parameters not included will be set to defaults. This will create one experiment and return it.
  - Example: Rounds = 2000, evolutionChance = 0.5:
    
    ``` experiment = fromParamDict({"rounds": 2000, "evolutionChance": 0.5})```
- Next, there is **parameterRange**. This is used to create multiple experiments sweeping across a parameter range. It only sweeps one parameter. Also, don't use this to sweep payoff matrices.
  This function takes four required parameters: **start**, **end**, **steps**, **param**. There is an optional **paramDict**.
  - **start**: Start value for the parameter sweep. Inclusive
  - **end**: End value for the parameter sweep. Inclusive
  - **steps**: Number of subdivisions
  - **param**: String name of parameter to sweep. See next section for more detail on these.
  - Optional **paramDict**: This allows you to set any of the other parameters not being swept to specified values. Allows you to tweak them from the default. Note! If you set the sweep parameter, it will be overwritten.
  - Example usage:

      ```experiments = builder.parameterRange(0,1,10,"evolutionChance")```

      ```experiments = builder.parameterRange(0,1,10,"evolutionChance", {"mutationRate": 0.1})```

- Finally, for sweeping payoff matrices, there is **payoffMatrixRange**. This has the same parameters as **parameterRange**, but instead of **param** it takes a tuple called **index**. Only sweeps 1 entry for the payoff matrix.
  - **index**: 2-tuple containing indices of which payoff matrix entry you would like to sweep. Each entry is either 0 or 1
    - (0,0): p00, (0,1): p01, (1,0): p10, (1,1): p11. See section 1 for definitions of these.
  - Example usage: 

      ```experiments = builder.payoffMatrixRange(3,4,10,(1,1))```
#### Running experiments
There are two commands you will need to run here. Referring to my example code, the first command to run once simulations are set up is ```builder.runAll()```. This is a method from the ExperimentBuilder class, and runs every experiment in the internal experiment list of the ExperimentBuilder. It takes no parameters. Next, ```builder.experimentList``` **overwrites** the text file ```experiments.txt``` with a list of the names of directories which the experiments it just ran output data into. This file will be used when displaying the data. That's all you need! If you want to run an experiment object by itself, just call ```run()``` on it with no parameters.

### Displaying data
There are three functions here, all essentially doing the same thing. They are not part of the experimentBuilder class!
- **fromPaths** is the fundamental one. It takes a list of Pathlib Path objects and displays them. I'll get to what it displays later. The path should contain the ```./Data```. For example, ```Data/exp0```. 
- **fromExperiments** takes a list of experiments, makes a list out of all their paths, and runs fromPaths with that.
- **fromTracker** takes no arguments. It reads a list of paths from the ```experiments.txt``` file and displays each of those.

What does it display? It uses two functions from the ```display.py``` file: ```displayAsImage()``` and ```heightmaps()```. DisplayAsImage is built for maxN=1, and it displays an animated image of the score each snapshot, and shows each rule value for an N=1 game on 4 animated images. Yellower = more likely to cooperate. heightmaps plots 5 3D histograms viewed from above: the score distribution and 4 histograms showing the rule values over time. The visible axes are snapshot number and the bins (either scores or probabilities). The unseen Z axis is the height of that bucket. The height is seen using the colormap: the whiter it is, the taller the bucket.

### Default parameters and parameter names
"gridN" = 32
"res" = (4,4)
"maxN" = 1
"rounds" = 10000
"iters" = 60
"snaps" = 100
"evolutionRate" = 0.01
"evolutionChance" = 0.2
"mutationRate" = 0.001
"payoffMatrix" = [[1,5],[0,3]]
"repeats" = 1
"gridSeed" = random.rand()*10000
"playSeed" = random.rand()*10000

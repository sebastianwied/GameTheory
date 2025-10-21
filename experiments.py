import numpy.random as random
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import os
from datetime import datetime
import csv
from display import displayAsImage, contourPlots

# compile with g++ -O3 -std=c++17 sim.cpp -o sim -pthread

######################################################
######### Handle data directory creation #############
######################################################
DATA_PATH = Path("./Data")
prefix = "exp"
def generateNewLogPath():
    # find all existing folders that match 'testN' pattern
    existing = [d for d in DATA_PATH.iterdir() if d.is_dir() and d.name.startswith(prefix)]

    # extract the numeric suffixes
    nums = []
    for d in existing:
        suffix = d.name[len(prefix):]
        if suffix.isdigit():
            nums.append(int(suffix))

    # pick the next available number
    next_num = max(nums, default=0) + 1
    new_folder = DATA_PATH / f"{prefix}{next_num}"
    
    print(f"Next folder will be: {new_folder}")
    # Optionally, actually create it:
    new_folder.mkdir(parents=True, exist_ok=False)
    return new_folder
######################################################

class Experiment:
    def __init__(self, execPath):
        ### Experiment parameters
        self.gridN = 32
        self.res = (4,4)
        self.maxN = 1
        self.rounds = 10000
        self.iters = 60
        self.snaps = 100
        self.evolutionRate = 0.01
        self.evolutionChance = 0.2
        self.mutationRate = 0.001
        self.payoffMatrix = [[1,5],[0,3]]
        self.repeats = 1
        self.seed1 = random.rand()*10000
        self.seed2 = random.rand()*10000
        
        ### Experiment execution
        self.execPath = execPath
        self.logPath = generateNewLogPath()
    
    def setParam(self, name, value):
        match name: 
            case "gridN":
                self.gridN = int(value)
            case "res":
                if type(value) != list and len(value) != 2:
                    print("bad resolution")
                    return
                self.res = value
            case "maxN":
                self.maxN = int(value)
            case "iters":
                self.iters = int(value)
            case "rounds":
                self.rounds = int(value)
            case "snaps":
                self.snaps = int(value)
            case "evolutionRate":
                self.evolutionRate = float(value)
            case "evolutionChance":
                self.evolutionChance = float(value)
            case "mutationRate":
                self.mutationRate = float(value)
            case "payoffMatrix":
                print(type(value))
                print(np.array(value).shape)
                if (type(value) not in [list, np.array]) and (np.array(value).shape != (2,2)):
                    print("Bad payoff matrix")
                    return
                self.payoffMatrix = value
            case "repeats":
                self.repeats = int(value)
            case "seed1":
                self.seed1 = int(value)
            case "seed2":
                self.seed2 = int(value)

    def run(self):
        for repeat in range(self.repeats):
            try: 
                subprocess.run(
                [self.execPath, str(self.payoffMatrix[0][0]), str(self.payoffMatrix[0][1]),
                str(self.payoffMatrix[1][0]), str(self.payoffMatrix[1][1]),
                str(self.gridN), str(self.res[0]), str(self.res[1]),
                str(self.maxN), str(self.rounds), str(self.iters),
                str(self.snaps), str(self.evolutionRate), str(self.mutationRate), 
                str(self.evolutionChance), str(self.seed1), str(self.seed2),
                str(self.logPath)]
                )
            except subprocess.CalledProcessError:
                pass
    
    def __repr__(self):
        return f"Payoff matrix: \n{self.payoffMatrix}, \nRounds: {self.rounds}, Iters: {self.iters}"

class ExperimentBuilder:
    def __init__(self, exec):
        self.execPath = exec
        self.experiments = []
    
    def new(self):
        return Experiment(self.execPath)

    def fromParamDict(self, paramDict):
        exp = self.new()
        for key, value in paramDict.items():
            exp.setParam(key, value)
        return exp
    
    def payoffMatrixRange(self, start, end, steps, index, paramDict=dict()):
        sweep = np.linspace(start, end, steps)
        base = np.array([[1,5],[0,3]], dtype = np.float64)
        baseMatrix = np.vstack([base]*len(sweep))
        baseMatrix = np.reshape(baseMatrix, (len(sweep),2,2))
        baseMatrix[:,index[0], index[1]] = sweep
        experiments = []
        for matrix in baseMatrix:
            exp = self.fromParamDict(paramDict)
            exp.setParam("payoffMatrix", matrix)
            experiments.append(exp)
            self.experiments.append(exp)
        return experiments

    def parameterRange(self, start, end, steps, param, paramDict=dict()):
        range = np.linspace(start, end, steps)
        experiments = []
        for val in range:
            exp = self.fromParamDict(paramDict)
            exp.setParam(param, val)
            experiments.append(exp)
            self.experiments.append(exp)
        return experiments
    
    def experimentList(self):
        with open("experiments.txt", "w") as f:
            f.truncate(0)
            for exp in self.experiments:
                f.write(f"{str(exp.logPath)}\n")
    
    def runAll(self):
        for exp in self.experiments:
            exp.run()

def load_csv(filename):
    """Load a CSV file and return a 2D numpy array."""
    data = np.loadtxt(filename, delimiter=',')
    return data

def fromTracker(tracker="experiments.txt"):
    paths = []
    with open(tracker, "r") as f:
        for line in f:
            paths.append(Path(line.strip()))
    fromPaths(paths)
    
def fromExperiments(exps):
    paths = []
    for exp in exps:
        paths.append(exp.logPath)
    fromPaths(paths)

def fromPaths(paths):
    for exp in paths:
        df = pd.read_csv(str(exp/Path("params.csv")), index_col=0)
        params = df['Value'].apply(pd.to_numeric).to_dict()
        snaps = int(params["snaps"])
        N = int(params["gridN"])
        maxN = int(params["maxN"])
        rounds = int(params["rounds"])
        matrix = [[int(params["p00"]),int(params["p01"])],[int(params["p10"]),int(params["p11"])]]
        scoreSnaps = load_csv(str(exp/Path("nonCumulativeScore.csv")))
        totalScore = load_csv(str(exp/Path("totalScore.csv")))
        ruleSnaps = load_csv(str(exp/Path("ruleSnaps.csv")))
        ruleSnaps = ruleSnaps.reshape(snaps, N, N, 4**maxN)
        scoreSnaps = scoreSnaps.reshape(snaps, N, N)
        displayAsImage(scoreSnaps, totalScore, ruleSnaps, matrix)
        contourPlots(scoreSnaps, totalScore, ruleSnaps, params["iters"])
pass

builder = ExperimentBuilder("./simTest")
exps = builder.payoffMatrixRange(3,4,5,(1,1))
#builder.runAll()
#builder.experimentList()
fromTracker()
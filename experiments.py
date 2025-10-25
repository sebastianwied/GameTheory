import numpy.random as random
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import os
from datetime import datetime
import csv
from display import displayAsImage, heightmaps, stateSpace4d

# compile with g++ -O3 -std=c++17 sim.cpp -o sim -pthread
#####################
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
        self.repeats = 1 ## TODO! When repeats != 1, it will overwrite the data from previous repeats. Fix!
        self.gridSeed = int(random.rand()*10000)
        self.playSeed = int(random.rand()*10000)
        self.varySeed = False
        self.inversionPercentage = 0
        self.inversionRound = self.rounds // 2
        
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
            case "gridSeed":
                self.gridSeed = int(value)
            case "playSeed":
                self.playSeed = int(value)
            case "varySeed":
                self.varySeed = bool(value)
            case "inversionPercentage":
                self.inversionPercentage = float(value)
            case "inversionRound":
                self.inversionRound = int(value)

    def run(self):
        self.saveParams()
        for repeat in range(self.repeats):
            subPath = self.logPath / Path(f"repeat{repeat}")
            subPath.mkdir(parents=True, exist_ok=False)
            if self.varySeed:
                self.seed1 = int(random.rand()*10000)
                self.seed2 = int(random.rand()*10000)
            try: 
                subprocess.run(
                [self.execPath, str(self.payoffMatrix[0][0]), str(self.payoffMatrix[0][1]),
                str(self.payoffMatrix[1][0]), str(self.payoffMatrix[1][1]),
                str(self.gridN), str(self.res[0]), str(self.res[1]),
                str(self.maxN), str(self.rounds), str(self.iters),
                str(self.snaps), str(self.evolutionRate), str(self.mutationRate), 
                str(self.evolutionChance), str(self.gridSeed), str(self.playSeed),
                str(subPath)]
                )
            except subprocess.CalledProcessError:
                pass
    
    def saveParams(self):
        with open(str(self.logPath/Path("params.csv")), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["p00", "p01","p10","p11","gridN","res0","res1","maxN", "rounds", "iters", "snaps", "evolutionRate", "evolutionChance", "mutationRate", "seed1", "seed2", "inversionPercentage", "inversionRound"])
            writer.writerow([
                self.payoffMatrix[0][0],self.payoffMatrix[0][1],self.payoffMatrix[1][0],self.payoffMatrix[1][1],
                self.gridN, self.res[0], self.res[1], self.maxN, self.rounds, self.iters, self.snaps,
                self.evolutionRate, self.evolutionChance, self.mutationRate, self.gridSeed, self.playSeed,
                self.inversionPercentage, self.inversionRound
                ])
    
    def __repr__(self):
        return f"Payoff matrix: \n{self.payoffMatrix}, \nRounds: {self.rounds}, Iters: {self.iters}"

class ExperimentBuilder:
    def __init__(self, exec):
        self.execPath = exec
        self.experiments = []
    
    def new(self):
        return Experiment(self.execPath)

    def fromParamDict(self, paramDict, add=True):
        exp = self.new()
        for key, value in paramDict.items():
            exp.setParam(key, value)
        if add==True:
            self.experiments.append(exp)
        return exp
    
    def payoffMatrixRange(self, start, end, steps, index, paramDict=dict()):
        sweep = np.linspace(start, end, steps)
        base = np.array([[1,5],[0,3]], dtype = np.float64)
        baseMatrix = np.vstack([base]*len(sweep))
        baseMatrix = np.reshape(baseMatrix, (len(sweep),2,2))
        baseMatrix[:,index[0], index[1]] = sweep
        experiments = []
        for matrix in baseMatrix:
            exp = self.fromParamDict(paramDict, add=False)
            exp.setParam("payoffMatrix", matrix)
            experiments.append(exp)
            self.experiments.append(exp)
        return experiments
    
    def resRange(self, start, end, paramDict=dict()):
        sweep = [start]
        while 2*sweep[-1][0] <= end[0]:
            sweep.append((2*sweep[-1][0],2*sweep[-1][1]))
        experiments = []
        for resolution in sweep:
            exp = self.fromParamDict(paramDict, add=False)
            exp.setParam("res", resolution)
            experiments.append(exp)
            self.experiments.append(exp)
        return experiments

    def parameterRange(self, start, end, steps, param, paramDict=dict(), varySeed=False):
        if param == "payoffMatrix":
            print("Use payoffMatrixRange. No experiments created")
            return
        range = np.linspace(start, end, steps)
        experiments = []
        gridSeed = random.rand()*10000
        playSeed = random.rand()*10000
        for val in range:
            if varySeed == True:
                gridSeed = random.rand()*10000
                playSeed = random.rand()*10000
            paramDict["gridSeed"] = gridSeed
            paramDict["playSeed"] = playSeed
            exp = self.fromParamDict(paramDict, add=False)
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
        dirs = [p for p in exp.iterdir() if p.is_dir()]
        df = pd.read_csv(str(exp / Path("params.csv")))
        params = df.iloc[0].to_dict()
        print(params)
        snaps = int(params["snaps"])
        N = int(params["gridN"])
        maxN = int(params["maxN"])
        rounds = int(params["rounds"])
        matrix = [[float(params["p00"]),float(params["p01"])],[float(params["p10"]),float(params["p11"])]]
        for dir in dirs:
            # Extract data
            scoreSnaps = load_csv(str(dir/Path("nonCumulativeScore.csv")))
            totalScore = load_csv(str(dir/Path("totalScore.csv")))
            ruleSnaps = load_csv(str(dir/Path("ruleSnaps.csv")))
            print(ruleSnaps.shape)
            print(snaps, N, maxN)
            ruleSnaps = ruleSnaps.reshape(snaps, N, N, 5)
            scoreSnaps = scoreSnaps.reshape(snaps, N, N)
            # Plot
            displayAsImage(scoreSnaps, totalScore, ruleSnaps, matrix)
            #heightmaps(scoreSnaps, totalScore, ruleSnaps, params["iters"])
            stateSpace4d(ruleSnaps)

def superPlot(tracker="experiments.txt"):
    path = ""
    with open(tracker, "r") as f:
        for line in f:
            path = Path(line.strip())
    dirs = [p for p in path.iterdir() if p.is_dir()]
    df = pd.read_csv(str(path / Path("params.csv")))
    params = df.iloc[0].to_dict()
    snaps = int(params["snaps"])
    N = int(params["gridN"])
    rounds = int(params["rounds"])
    matrix = [[float(params["p00"]),float(params["p01"])],[float(params["p10"]),float(params["p11"])]]
    finalRules = []
    for dir in dirs:
        # Extract data
        ruleSnaps = load_csv(str(dir/Path("ruleSnaps.csv")))
        ruleSnaps = ruleSnaps.reshape(snaps, N, N, 5)
        finalRules.append(ruleSnaps[-1])
    finalRules = np.array(finalRules)
    print(finalRules.shape)
    stateSpace4d(np.array(finalRules))


builder = ExperimentBuilder("./sim")
paramdict = {"repeats": 1, "rounds": 10000, "snaps": 200, "gridN": 128, "varySeed": False, 
                            "payoffMatrix": [[1,5],[0,3.3]], "inversionPercentage": 0.1,
                            "mutationRate": 0.005, "res": (2,2)}
exp = builder.fromParamDict(paramdict)
#builder.resRange((1,1), (16,16), paramdict)
#builder.fromParamDict(paramdict)
builder.runAll()
builder.experimentList()
fromTracker()
#superPlot()
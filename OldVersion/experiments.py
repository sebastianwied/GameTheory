import numpy.random as random
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import os
from datetime import datetime
import csv
from display import displayAsImage, heightmaps, stateSpace4d
from Perlin_numpy import *
import matplotlib.pyplot as plt

# compile with g++ -O3 -std=c++17 simNewMapGen.cpp -o sim -pthread
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

################################################################################
# Map Generation ###############################################################
################################################################################

PERLIN = 0
GAUSSIAN = 1
CONSTANT = 2

class MapObject:
    def __init__(self):
        self.value = 0
        self.seed = None
        self.resolution = (0,0)
        self.sigmas = (0,0)
        self.span = (0,0,0,0) # (xMin, xMax, yMin, yMax)
        self.amplitude = 0
        self.truncation = 0
        self.norm = False
        self.constructed = False

class GaussianMap(MapObject):
    type = GAUSSIAN
    def __init__(self, sigmas, span, amplitude, truncation, norm):
        super().__init__()
        self.sigmas = sigmas
        self.span = span # (xMin, xMax, yMin, yMax)
        self.amplitude = amplitude
        self.truncation = truncation
        self.norm = norm
        self.seed = None
    
    def constructMap(self, gridN, gridSeed):  #gridSeed not used, only here for easier calling code later
        xR = np.linspace(self.span[0], self.span[1], gridN)
        yR = np.linspace(self.span[2], self.span[3], gridN)
        #print(xR.shape, yR.shape)
        def value(x, y):
            return self.amplitude*np.exp((-x*x/(2*self.sigmas[0]*self.sigmas[0]))+(-y*y/(2*self.sigmas[1]*self.sigmas[1])))
        grid = np.zeros((gridN, gridN))
        for idx in range(gridN):
            for idy in range(gridN):
                #print(idx, idy)
                grid[idy, idx] = min(value(xR[idx], yR[-(idy+1)]), self.truncation)
        if self.norm: grid /= np.nanmax(grid)
        self.map = grid
        self.constructed = True
        return self.map

class PerlinMap(MapObject):
    type = PERLIN
    def __init__(self, resolution, seed=None, mean=None, precision=None):
        super().__init__()
        self.resolution = resolution
        self.seed = seed
        self.mean = mean
        self.precision = precision
        if mean != None and precision == None:
            print("Precision should not be None")
            self.precision = 0.1
    
    def shiftMean(self, target, precision):
        print("Shifting mean")
        newMap = np.copy(self.map)
        mean = np.mean(newMap)
        counter = 0
        while (mean > target + precision) or (mean < target - precision):
            #print(f"diff: {target - mean}")
            newMap += (target - mean) / 10
            #newMap /= np.nanmax(newMap)
            mean = np.mean(newMap)
            #print(mean)
            counter += 1
            if counter > 50:
                break
        self.map = newMap
        self.constructed = True
        
    def constructMap(self, gridN, gridSeed):
        map = generate_fractal_noise_2d((gridN, gridN), self.resolution, seed=gridSeed)
        map += 1
        map /= 2
        self.map = map
        if self.mean != None and self.precision != None:
            self.shiftMean(self.mean, self.precision)
        self.constructed = True
        return map

class ConstantMap(MapObject):
    type = CONSTANT
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.seed = None
    
    def constructMap(self, gridN, gridSeed): #gridSeed not used, only here for easier calling code later
        self.map = np.ones((gridN, gridN)) * self.value
        self.constructed = True
        return self.map

################################################################################
# Experiment functionality #####################################################
################################################################################

class Experiment:
    def __init__(self, execPath):
        ### Experiment parameters
        self.gridN = 32
        self.maxN = 1
        self.rounds = 10000
        self.iters = 60
        self.snaps = 100
        self.evolutionRate = 0.01
        self.evolutionChance = 0.2
        self.mutationRate = 0.001
        self.payoffMatrix = [[1,5],[0,3]]
        self.repeats = 1
        self.playSeed = int(random.rand()*10000)
        self.gridSeed = int(random.rand()*10000)
        self.varySeed = False
        
        ## Maps
        self.defaultRes = (2,2)
        self.mapRoots = [PerlinMap(self.defaultRes), # DD
                    PerlinMap(self.defaultRes), # DC
                    PerlinMap(self.defaultRes), # CD
                    PerlinMap(self.defaultRes), # CC
                    ConstantMap(self.mutationRate)] # Mutation Rate
        
        ### Experiment execution
        self.execPath = execPath
        self.logPath = generateNewLogPath()
    
    def setParam(self, name, value):
        match name: 
            case "gridN":
                self.gridN = int(value)
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
            case "payoffMatrix":
                print(type(value))
                print(np.array(value).shape)
                if (type(value) not in [list, np.array]) and (np.array(value).shape != (2,2)):
                    print("Bad payoff matrix")
                    return
                self.payoffMatrix = value
            case "repeats":
                self.repeats = int(value)
            case "playSeed":
                self.playSeed = int(value)
            case "varySeed":
                self.varySeed = bool(value)
            case "gridSeed":
                self.gridSeed = int(value)
            case "maps":
                self.mapRoots = value

    def constructMaps(self):
        ## Generate map seeds
        random.seed(self.gridSeed)
        seeds = random.random(5)
        ##
        maps = self.mapRoots
        if len(maps) != 5:
            print("Wrong number of maps. Keeping default")
            return
        ## Generate maps
        self.maps = np.empty((5,self.gridN, self.gridN))
        self.mapTypes = []
        self.mapSeeds = []
        count = 0
        for seed, map in zip(seeds,maps):
            seed = int(seed*10000)
            if map.seed == None:
                # Does not contain predefined seed
                if map.constructed:
                    grid = map.map
                else:
                    map.constructMap(self.gridN, map.seed)
                    grid = map.map
                self.maps[count,:,:] = grid
                self.mapSeeds.append(seed)
                self.mapTypes.append(map.type)
            else:
                # Contains predefined seed.
                if map.constructed:
                    grid = map.map
                else:
                    map.constructMap(self.gridN, map.seed)
                    grid = map.map
                self.maps[count,:,:] = grid
                self.mapTypes.append(map.type)
                self.mapSeeds.append(map.seed)
            self.mapTypes.append(map.type)
            count += 1
        ##

    def writeMaps(self, path):
        # Wrie to ruleMaps file. simulation reads this file and uses it as the map.
        # Structure: gridN lines for each map, gridN many entries in that line.
        with open(str(path / Path("maps.csv")), 'w') as f:
            writer = csv.writer(f)
            print(self.maps.shape)
            for map in self.maps:
                print(map.shape)
                for row in map:
                    writer.writerow(row)
        pass

    def run(self):
        for repeat in range(self.repeats):
            self.constructMaps()
            subPath = self.logPath / Path(f"repeat{repeat}")
            subPath.mkdir(parents=True, exist_ok=False)
            self.writeMaps(subPath)
            try: 
                subprocess.run(
                [self.execPath, str(self.payoffMatrix[0][0]), str(self.payoffMatrix[0][1]),
                str(self.payoffMatrix[1][0]), str(self.payoffMatrix[1][1]),
                str(self.gridN), str(self.rounds), str(self.iters),
                str(self.snaps), str(self.evolutionRate),
                str(self.evolutionChance), str(self.playSeed),
                str(subPath)]
                )
            except subprocess.CalledProcessError:
                pass
            self.saveParams(subPath)
            if self.varySeed:
                self.gridSeed = int(random.rand()*10000)
                self.playSeed = int(random.rand()*10000)

    def saveParams(self, path):
        with open(str(path/Path("params.csv")), "w") as f:
            writer = csv.writer(f)
            labels = ["p00", "p01", "p10", "p11", "gridN", "rounds", "iters", "snaps", "evolutionRate", "evolutionChance", "mutationRate", "gridSeed", "playSeed"]
            entries = [
            self.payoffMatrix[0][0],self.payoffMatrix[0][1],self.payoffMatrix[1][0],self.payoffMatrix[1][1], self.gridN, self.rounds, 
            self.iters, self.snaps, self.evolutionRate, self.evolutionChance, self.mutationRate, self.gridSeed, self.playSeed]
            writer.writerow(labels)
            writer.writerow(entries)
        # Write map info
        pathNames = [Path("map00.csv"),Path("map01.csv"),Path("map10.csv"),Path("map11.csv")]
        for mapPath, map, seed in zip(pathNames, self.mapRoots[:4], self.mapSeeds[:4]):
            with open(str(path/mapPath), "w") as f:
                writer = csv.writer(f)
                labels = ["type", "seed", "res0", "res1", "value", "sigmaX", "sigmaY", "xMin", "xMax", "yMin", "yMax", "amplitude", "truncation", "norm"]
                entries = [map.type, seed, map.resolution[0], map.resolution[1], map.value, map.sigmas[0], map.sigmas[1], map.span[0], map.span[1], map.span[2], map.span[3], map.amplitude, map.truncation, map.norm]
                writer.writerow(labels)
                writer.writerow(entries)
    
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

class ParamCarrier:
    def __init__(self, path):
        df = pd.read_csv(str(path/Path("params.csv")))
        self.simParams = df.iloc[0].to_dict()
        df = pd.read_csv(str(path/Path("map00.csv")))
        self.m00 = df.iloc[0].to_dict()
        df = pd.read_csv(str(path/Path("map01.csv")))
        self.m01 = df.iloc[0].to_dict()
        df = pd.read_csv(str(path/Path("map10.csv")))
        self.m10 = df.iloc[0].to_dict()
        df = pd.read_csv(str(path/Path("map11.csv")))
        self.m11 = df.iloc[0].to_dict()

def fromPaths(paths):
    for exp in paths:
        dirs = [p for p in exp.iterdir() if p.is_dir()]
        for dir in dirs:
            params = ParamCarrier(dir)
            snaps = int(params.simParams["snaps"])
            N = int(params.simParams["gridN"])
            # Extract data
            scoreSnaps = load_csv(str(dir/Path("nonCumulativeScore.csv")))
            totalScore = load_csv(str(dir/Path("totalScore.csv")))
            ruleSnaps = load_csv(str(dir/Path("ruleSnaps.csv")))
            ruleSnaps = ruleSnaps.reshape(snaps, N, N, 5)
            scoreSnaps = scoreSnaps.reshape(snaps, N, N)
            # Plot
            displayAsImage(scoreSnaps, totalScore, ruleSnaps, params)
            #heightmaps(scoreSnaps, totalScore, ruleSnaps, params["iters"])
            #stateSpace4d(ruleSnaps)

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
    stateSpace4d(np.array(finalRules))

builder = ExperimentBuilder("./sim")
res = (2,2)
gridN = 64
sigmas = (2,2)
shape = (-4,4,-4,4)
amplitude = 1
gridSeed = int(random.rand()*10000)
maps = np.array([
    PerlinMap(res, mean=0.4, precision=0.01),
    PerlinMap(res, mean=0.4, precision=0.01),
    GaussianMap(sigmas, shape, amplitude, truncation = 2, norm=False),
    PerlinMap(res, mean=0.3, precision=0.01),
    ConstantMap(0.01)
])

paramdict = {"repeats": 1, "rounds": 10000, "snaps": 100, "varySeed": True,
            "gridSeed": 9163, "playSeed": 3288,
            "payoffMatrix": [[1,5],[0,3]],"evolutionChance": 0.5, 
            "evolutionRate":0.2, "gridN":gridN, "maps":maps}
builder.fromParamDict(paramdict)
builder.runAll()
builder.experimentList()
fromTracker()
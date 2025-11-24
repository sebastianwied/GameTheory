// sim.cpp
// Single-file C++ port of the Python code you supplied.
// Compile: g++ -O3 -std=c++17 sim.cpp -o sim -pthread
//
// Outputs CSV files:
//  - scoreSnaps.csv (snapshots of cumulative score at snapshot times)
//  - totalScore.csv (final total scores grid)
//  - ruleSnaps.csv (flattened rule snapshots: snap, y, x, ruleIndex, value)
//  - nonCumulativeScore.csv (snapshots of scoreTracker per snap)
//
#define NOMINMAX
#include <fstream>
#include <thread>
#include <random>
#include <cmath>       // if you use math functions
#include <chrono>      // if you use std::chrono for timing
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <queue>
#include <stack>
#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <memory>
#include <sstream>
using namespace std;

/* ---------------------------
Utility / random helpers
--------------------------- */

using u64 = unsigned long long;
std::mt19937_64 grid_rng;
std::mt19937_64 global_rng;

double uniform01() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(global_rng);
}

int randint(int a, int b) { // inclusive [a,b]
    return std::uniform_int_distribution<int>(a,b)(global_rng);
}

/* ---------------------------
Perlin / Fractal noise
Ported from your perlin_numpy implementation
--------------------------- */

/* ---------------------------
MemoryN classes (agents)
--------------------------- */

enum Move: int {
    COOP = 1,
    DEF = 0
};

struct Memory1 {
    Move startMove;
    Move prevMove;
    array<double,4> rule;
    double score;
    string name;
    double mutationRate;

    Memory1(Move _start = COOP, double _mutationRate=0.01):
        startMove(_start),
        prevMove(startMove),
        rule{0.0,0.0,0.0,0.0},
        mutationRate(_mutationRate),
        score(0.0),
        name("MemoryN")
    {}

    virtual void startup(Move _start) {
        this->startMove = _start;
        this->prevMove = _start;
        this->score = 0.0;
    }

    int playMove(int theirPrev, double seed, int roundNum) {
        // if (roundNum == 0) {
        //     prevMove = startMove;
        //     return startMove;
        // } Omit in favor of doing this check when running the individual game, rather than at every call to playMove
        int key = (prevMove << 1) | theirPrev;
        double prob = rule[key];
        if (seed < prob) {
            prevMove = COOP;
            return COOP;
        } else {
            prevMove = DEF;
            return DEF;
        }
    }

    void reset() {
        score = 0.0;
        prevMove = startMove;
    }

    virtual string repr() const {
        return name;
    }

    void setRule(const array<double, 4> &r) {
        rule[0] = r[0];
        rule[1] = r[1];
        rule[2] = r[2];
        rule[3] = r[3];
    }
};

struct BLANK : public Memory1 {
    BLANK(Move start = COOP, double _mutationRate=0.01) : Memory1(start) {
        name = "BLANK";
        rule[0] = 0.0;
        rule[1] = 0.0;
        rule[2] = 0.0;
        rule[3] = 0.0;
        startMove = start;
        prevMove = start;
        mutationRate = _mutationRate;
    }
};

/* ---------------------------
Grid generation
--------------------------- */

using AgentGrid = vector<vector<shared_ptr<Memory1>>>;

std::vector<std::vector<std::vector<double>>>
read_blocks_csv(const std::string &filename, int N) {

    std::vector<std::vector<std::vector<double>>> data(
        5, std::vector<std::vector<double>>(N, std::vector<double>(N))
    );

    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::string line;
    int block = 0;
    int row_in_block = 0;

    while (std::getline(file, line)) {
        // skip empty lines if they appear
        if (line.size() == 0) continue;

        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (std::getline(ss, cell, ',')) {
            data[block][row_in_block][col] = std::stod(cell);
            col++;
        }

        row_in_block++;

        // move to next block if needed
        if (row_in_block == N) {
            block++;
            row_in_block = 0;
            if (block == 5) break; // done
        }
    }

    return data;
}

// blankGrid(N, res, maxN=1, seed)
AgentGrid importGrid(int N, const std::string &filename) 
    {
    AgentGrid grid(N, vector<shared_ptr<Memory1>>(N));
    auto paramMaps = read_blocks_csv(filename, N);

    for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        // create rule vector of length 4. The python did: setRule([i**2 for i in list(paramMaps[:,idr,idc])])
        // paramMaps[:,idr,idc] is "number" values — they square them.
        array<double,4> ruleVals;
        // If number (channels) does not equal rule size, we'll distribute or repeat
        // but the Python used list(paramMaps[:,idr,idc]) and squared those values to form rule,
        // meaning rule length == number. But rule length should be 4**maxN. In the python blankGrid,
        // they use number=(4**maxN) so number == rule length. So here it's consistent.
        for (int k=0;k<4;++k) {
            double v = paramMaps[k][i][j];
            ruleVals[k] = v;
            // clamp [0,1]
            if (ruleVals[k] < 0.0) ruleVals[k] = 0.0;
            if (ruleVals[k] > 1.0) ruleVals[k] = 1.0;
        }
        auto ag = make_shared<BLANK>(COOP, paramMaps[4][i][j]);
        ag->setRule(ruleVals);
        grid[i][j] = ag;
    }

    return grid;
}

/* ---------------------------
Tournaments
--------------------------- */

static vector<vector<double>> payoffMatrix = {{1,5},{0,3}};

struct TorusResult {
    // snapshots: vector of 2D arrays (snap index -> N x N)
    vector<vector<vector<double>>> scoreSnaps; // snap -> y -> x
    vector<vector<vector<double>>> ruleSnaps;  // snap -> y -> x*ruleLen (flattened per x)
    vector<vector<vector<double>>> nonCumulativeScoreSnaps; // snap -> y -> x
    vector<vector<double>> totalScore; // y x
};

int flatten_index(int y, int x, int X) { return y*X + x; }

vector<vector<pair<int,int>>> pickOpponents(const AgentGrid &agentGrid) {
    int yLen = (int)agentGrid.size();
    int xLen = (int)agentGrid[0].size();
    int N = yLen * xLen;
    vector<double> angles(N);
    for (int i=0;i<N;++i) angles[i] = uniform01() * 2.0 * M_PI;
    vector<int> xs(N), ys(N);
    for (int i=0;i<N;++i) {
        xs[i] = (int)round(cos(angles[i]));
        ys[i] = (int)round(sin(angles[i]));
    }
    vector<vector<pair<int,int>>> opponent(yLen, vector<pair<int,int>>(xLen));
    for (int iy=0; iy<yLen; ++iy) {
        for (int ix=0; ix<xLen; ++ix) {
            int id = iy * xLen + ix;
            int xLoc = (ix + xs[id]) % xLen;
            if (xLoc < 0) xLoc += xLen;
            int yLoc = (iy + ys[id]) % yLen;
            if (yLoc < 0) yLoc += yLen;
            opponent[iy][ix] = {xLoc, yLoc};
        }
    }
    return opponent;
}

static const pair<int,int> DIRS[8] = {
    { 1, 0}, {-1, 0}, {0, 1}, {0,-1},
    { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1}
};

vector<vector<pair<int,int>>> pickOpponentsNew(const AgentGrid &agents) {
    int Y = agents.size();
    int X = agents[0].size();

    vector<vector<pair<int,int>>> opp(Y, vector<pair<int,int>>(X));

    for(int y=0; y<Y; ++y) {
        for(int x=0; x<X; ++x) {
            int d = rand() % 8;
            int nx = (x + DIRS[d].first + X) % X;
            int ny = (y + DIRS[d].second + Y) % Y;
            opp[y][x] = {nx, ny};
        }
    }
    return opp;
}

vector<vector<array<double,5>>> agentRuleSnapshot(const AgentGrid &agents) {
    int Y = agents.size();
    int X = agents[0].size();

    vector<vector<array<double,5>>> snap(Y, vector<array<double,5>>(X));
    for(int i=0; i<Y; ++i){
        for(int j=0; j<X; ++j) {
            for(int k=0; k<4; ++k) snap[i][j][k] = agents[i][j]->rule[k];
            snap[i][j][4] = agents[i][j]->mutationRate;
        }
    }
    return snap;
}

TorusResult torusTournament(AgentGrid agentGrid, int iters, int rounds, int snaps, float evolutionRate, 
    float evolutionChance) {

    int yLen = (int)agentGrid.size();
    int xLen = (int)agentGrid[0].size();
    int N = yLen * xLen;
    TorusResult out;
    vector<vector<double>> totalScore(yLen, vector<double>(xLen, 0.0));
    int snapEvery = max(1, rounds / snaps);
    vector<vector<double>> paddedScore(yLen+2, vector<double>(xLen+2, 0.0));

    for (int round=0; round<rounds; ++round) {
        // PLAY MATCHES
        auto matchups = pickOpponents(agentGrid);
        vector<vector<int>> playedTracker(yLen, vector<int>(xLen, 0));
        vector<vector<double>> scoreTracker(yLen, vector<double>(xLen, 0.0));

        // BEFORE launching threads: create deterministic thread seeds and decide nThreads
        int nThreads = std::min(static_cast<int>(std::thread::hardware_concurrency()), (int) yLen);
        if (nThreads < 1) nThreads = 1;

        // Create thread seeds deterministically using global_rng (seeded in main)
        vector<uint64_t> thread_seeds(nThreads);
        for (int t = 0; t < nThreads; ++t) {
            thread_seeds[t] = global_rng(); // deterministic sequence
        }

        // Prepare per-thread accumulators
        vector<vector<vector<double>>> scoreTracker_threads(nThreads,
            vector<vector<double>>(yLen, vector<double>(xLen, 0.0)));
        vector<vector<vector<int>>> playedTracker_threads(nThreads,
            vector<vector<int>>(yLen, vector<int>(xLen, 0)));

        // Worker now receives thread id and seed; uses its own local RNG
        auto worker = [&](int t_id, int startRow, int endRow) {
            std::mt19937_64 local_rng(thread_seeds[t_id]);
            std::uniform_real_distribution<double> unif(0.0, 1.0);

            auto local_uniform01 = [&](){ return unif(local_rng); };

            // local references to thread-local accumulators
            auto &scoreTracker_local = scoreTracker_threads[t_id];
            auto &playedTracker_local = playedTracker_threads[t_id];

            for (int idy = startRow; idy < endRow; ++idy) {
                for (int idx = 0; idx < xLen; ++idx) {
                    auto match = matchups[idy][idx];
                    auto a1 = agentGrid[idy][idx];
                    auto a2 = agentGrid[match.second][match.first];

                    // increment played count for both players in THREAD-LOCAL arrays
                    playedTracker_local[idy][idx] += 1;
                    playedTracker_local[match.second][match.first] += 1;

                    // generate seeds for iterated plays using local_rng
                    vector<double> seeds(2 * iters);
                    for (int s = 0; s < 2*iters; ++s) seeds[s] = local_uniform01();

                    for (int n = 0; n < iters; ++n) {
                        unsigned long long a1prev = a1->prevMove;
                        unsigned long long a2prev = a2->prevMove;
                        int a1move = a1->playMove(a2prev, seeds[n], n);
                        int a2move = a2->playMove(a1prev, seeds[n+iters], n);
                        // accumulate into thread-local arrays
                        scoreTracker_local[idy][idx] += payoffMatrix[a1move][a2move];
                        scoreTracker_local[match.second][match.first] += payoffMatrix[a2move][a1move];
                    }
                    a1->reset();
                    a2->reset();
                }
            }
        };


        int rowsPerThread = std::max(1, ( (int) yLen) / nThreads);
        int row = 0;
        vector<thread> threads;
        for (int t = 0; t < nThreads; ++t) {
            int startR = row;
            int endR = std::min((int) yLen, row + rowsPerThread);
            if (t == nThreads - 1) endR = yLen;
            threads.emplace_back(worker, t, startR, endR);
            row = endR;
        }
        for (auto &th : threads) if (th.joinable()) th.join();
        threads.clear();

        // zero out global trackers then sum thread-local results deterministically
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) {
            playedTracker[i][j] = 0;
            scoreTracker[i][j] = 0.0;
        }

        for (int t=0; t<nThreads; ++t) {
            for (int i=0;i<yLen;++i) {
                for (int j=0;j<xLen;++j) {
                    playedTracker[i][j] += playedTracker_threads[t][i][j];
                    scoreTracker[i][j] += scoreTracker_threads[t][i][j];
                }
            }
        }

        // Normalize by playedTracker (avoid div by zero)
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) {
            if (playedTracker[i][j] > 0) scoreTracker[i][j] /= (double)playedTracker[i][j];
            totalScore[i][j] += scoreTracker[i][j];
        }

        // Evolution
        AgentGrid newGrid = agentGrid; // shallow copy of shared_ptrs
        double shiftPercentage = evolutionRate;
        // build padded score toroidally
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) paddedScore[i+1][j+1] = scoreTracker[i][j];
        // wrap edges
        for (int j=0;j<xLen;++j) paddedScore[0][j+1] = scoreTracker[yLen-1][j];
        for (int j=0;j<xLen;++j) paddedScore[yLen+1][j+1] = scoreTracker[0][j];
        for (int i=0;i<yLen;++i) paddedScore[i+1][0] = scoreTracker[i][xLen-1];
        for (int i=0;i<yLen;++i) paddedScore[i+1][xLen+1] = scoreTracker[i][0];
        paddedScore[0][0] = scoreTracker[yLen-1][xLen-1];
        paddedScore[yLen+1][xLen+1] = scoreTracker[0][0];
        paddedScore[0][xLen+1] = scoreTracker[yLen-1][0];
        paddedScore[yLen+1][0] = scoreTracker[0][xLen-1];

        // decide evolution for each agent
        vector<double> evolveVec(N);
        for (int i=0;i<N;++i) evolveVec[i] = uniform01();
        double chance = 0.1;
        int count = -1;
        for (int idy=0; idy<yLen; ++idy) {
            for (int idx=0; idx<xLen; ++idx) {
                ++count;
                // find max in local 3x3 window
                int yMin = idy;
                int yMax = idy+3;
                int xMin = idx;
                int xMax = idx+3;
                // padded region is paddedScore[yMin:yMax, xMin:xMax] (size 3x3)
                double mx = -1e300;
                int bestIndex = 0;
                for (int py=yMin; py<yMax; ++py) for (int px=xMin; px<xMax; ++px) {
                    double val = paddedScore[py][px];
                    int linear = (py - yMin) * 3 + (px - xMin);
                    if (val > mx) { mx = val; bestIndex = linear; }
                }
                // map bestIndex to neighbor coords (unwrapped)
                int uy = ((bestIndex / 3) - 1) + idy;
                int ux = ((bestIndex % 3) - 1) + idx;
                // wrap
                if (uy < 0) uy = yLen + uy;
                else if (uy >= yLen) uy -= 1;
                if (ux < 0) ux = xLen + ux;
                else if (ux >= xLen) ux -= 1;

                // shift
                vector<double> ruleShift(agentGrid[idy][idx]->rule.size(), 0.0);
                if (evolveVec[count] < chance) {
                    auto &src = agentGrid[uy][ux]->rule;
                    auto &dst = agentGrid[idy][idx]->rule;
                    for (size_t k=0;k<dst.size();++k) ruleShift[k] = (src[k] - dst[k]) * shiftPercentage;
                }
                // mutate
                vector<double> ruleShift2(agentGrid[idy][idx]->rule.size(), 0.0);
                for (size_t k=0;k<ruleShift2.size();++k) {
                    ruleShift2[k] = ((uniform01() * 2.0) - 1.0) * (agentGrid[idy][idx]->mutationRate);
                }
                array<double,4> newRule = agentGrid[idy][idx]->rule;
                for (size_t k=0;k<newRule.size();++k) newRule[k] = newRule[k] + ruleShift[k] + ruleShift2[k];
                for (size_t k=0;k<newRule.size();++k) {
                    if (newRule[k] < 0.0) newRule[k] = 0.0;
                    if (newRule[k] > 1.0) newRule[k] = 1.0;
                }
                // assign to newGrid copy
                // make a fresh BLANK agent to hold new rule while preserving other meta
                auto newAgent = make_shared<BLANK>(agentGrid[idy][idx]->startMove);
                newAgent->name = agentGrid[idy][idx]->name;
                newAgent->rule = newRule;
                newAgent->startMove = agentGrid[idy][idx]->startMove;
                newAgent->prevMove = agentGrid[idy][idx]->prevMove;
                newGrid[idy][idx] = newAgent;
            }
        }
        agentGrid = newGrid;

        if (round == 1 || ((round % snapEvery) == 0 && (round / snapEvery) > 0)) {
            // push snapshots
            out.scoreSnaps.push_back(totalScore);
            auto ruleSnap = agentRuleSnapshot(agentGrid);
            // For storage simplicity, push ruleSnaps as flattened vectors per cell
            // but we'll convert to vector<vector<vector<double>>> where innermost is concatenated rule vector per cell
            size_t ruleLen = 5;//(int)agentGrid[0][0]->rule.size();
            // flatten rules into 2D matrix of (y, x*ruleLen) to mimic original
            vector<vector<double>> flatRules(yLen, vector<double>(xLen * ruleLen));
            for (int iy=0; iy<yLen; ++iy) for (int ix=0; ix<xLen; ++ix) {
                for (int k=0;k<ruleLen;++k) flatRules[iy][ix*ruleLen + k] = ruleSnap[iy][ix][k];
            }
            // store flatRules into ruleSnaps (but as 3D: snap -> y -> x*ruleLen)
            out.ruleSnaps.push_back(flatRules);
            out.nonCumulativeScoreSnaps.push_back(scoreTracker);
            cout << "progress: " << (round / snapEvery) << " / "<<snaps<<"\n";
        }
    } // end rounds

    out.totalScore = totalScore;
    return out;
}

/* ---------------------------
Output helpers (CSV)
--------------------------- */

void write_scoreSnaps_csv(const vector<vector<vector<double>>> &scoreSnaps, const string &fname) {
    std::cout << "Writing scoreSnaps to: " << fname << std::endl;
    ofstream f(fname);
    // write each snap as flattened row; comment header
    for (size_t s=0; s<scoreSnaps.size(); ++s) {
        auto &grid = scoreSnaps[s];
        int H = grid.size(), W = grid[0].size();
        // flatten
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                f << grid[i][j];
                if (!(i==H-1 && j==W-1)) f << ",";
            }
        }
        f << "\n";
    }
    f.close();
}

void write_totalScore_csv(const vector<vector<double>> &totalScore, const string &fname) {
    std::cout << "Writing totalScore to: " << fname << std::endl;
    ofstream f(fname);
    int H = totalScore.size(), W = totalScore[0].size();
    for (int i=0;i<H;++i) {
        for (int j=0;j<W;++j) {
            f << totalScore[i][j];
            if (j < W-1) f << ",";
        }
        f << "\n";
    }
    f.close();
}


void write_ruleSnaps_csv(const vector<vector<vector<double>>> &ruleSnaps, const string &fname) {
    std::cout << "Writing ruleSnapss to: " << fname << std::endl;
    ofstream f(fname);
    // Each line: snap_index,y,x,ruleIndex,ruleValue   (sparse long form)
    int snaps = ruleSnaps.size();
    // For each snap
    for (int s=0; s<snaps; ++s) {
        //For each grid flatRules
        auto &flatRules = ruleSnaps[s]; // y -> x*ruleLen
        int y = (int)flatRules.size();
        int Xflat = (int)flatRules[0].size();
        // we don't know ruleLen directly; but it's Xflat / xLen. To keep things simple, output flattened full lines:
        // for each row write all values as a long comma-separated line (snap per line)
        // For row I
        for (int i=0;i<y;++i) {
            // for column J
            for (int j=0;j<Xflat;++j) {
                f << to_string(flatRules[i][j]);
                if (j<Xflat-1) f << ",";
            }
            f << "\n";
        }
    }
    f.close();
}

void write_nonCumulative_csv(const vector<vector<vector<double>>> &ncs, const string &fname) {
    std::cout << "Writing nonCumulativeScore to: " << fname << std::endl;
    ofstream f(fname);
    for (size_t s=0;s<ncs.size();++s) {
        auto &grid = ncs[s];
        int H = grid.size(), W = grid[0].size();
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                f << grid[i][j];
                if (!(i==H-1 && j==W-1)) f << ",";
            }
        }
        f << "\n";
    }
    f.close();
}

/* ---------------------------
main (testing)
--------------------------- */

int main(int argc, char** argv) {
    payoffMatrix = {
        {atof(argv[1]), atof(argv[2])},
        {atof(argv[3]), atof(argv[4])}
    };

    int gridN = atoi(argv[5]);
    int rounds = atoi(argv[6]);
    int iters = atoi(argv[7]);
    int snaps = atoi(argv[8]);
    double evolutionRate = atof(argv[9]);
    double evolutionChance = atof(argv[10]);
    unsigned int playSeed = (unsigned) std::atoi(argv[11]);
    global_rng.seed(playSeed);

    std::string path = argv[12];

    cout << path;

    cout << "Building grid...\n";
    AgentGrid grid = importGrid(gridN, path+"/maps.csv");

    cout << "Running tournament (" << rounds << " rounds, " << iters << " iters per match)...\n";
    TorusResult resu = torusTournament(grid, iters, rounds, snaps, evolutionRate, evolutionChance);

    write_scoreSnaps_csv(resu.scoreSnaps, path+"/scoreSnaps.csv");
    write_totalScore_csv(resu.totalScore, path+"/totalScore.csv");
    write_ruleSnaps_csv(resu.ruleSnaps, path+"/ruleSnaps.csv");
    write_nonCumulative_csv(resu.nonCumulativeScoreSnaps, path+"/nonCumulativeScore.csv");

    return 0;
}
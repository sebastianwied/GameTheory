// sim.cpp
// Single-file C++ port of the Python code you supplied.
// Compile: g++ -O3 -std=c++17 sim.cpp -o sim -pthread
// Run: ./sim
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
#include <vector>
#include <random>
#include <cmath>       // if you use math functions
#include <chrono>      // if you use std::chrono for timing
#include <iostream>
#include <cstdio>
#include <cmath>
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
using namespace std;

/* ---------------------------
   Utility / random helpers
   --------------------------- */

using u64 = unsigned long long;
std::mt19937_64 global_rng;
std::mt19937_64 grid_rng;

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

static inline double interpolant(double t) {
    // t * t * t * (t * (t * 6 - 15) + 10)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Helper: create 3D array-like vectors: shape: (n, h, w)
using Noise3D = vector<vector<vector<double>>>;

// generate_perlin_noise_2d(shape, res, tileable, seed, number)
Noise3D generate_perlin_noise_2d(pair<int,int> shape, pair<int,int> res, pair<bool,bool> tileable, unsigned seed, int number=1) {
    int H = shape.first;
    int W = shape.second;
    Noise3D data(number, vector<vector<double>>(H, vector<double>(W, 0.0)));

    for (int n=0;n<number;++n) {
        double delta0 = (double)res.first / (double)H;
        double delta1 = (double)res.second / (double)W;
        int d0 = H / res.first;
        int d1 = W / res.second;

        // grid coords
        // grid[i][j] = (u, v) where u and v in [0,1)
        vector<vector<pair<double,double>>> grid(H, vector<pair<double,double>>(W));
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                double gx = fmod((i * delta0), 1.0);
                double gy = fmod((j * delta1), 1.0);
                grid[i][j] = {gx, gy};
            }
        }

        // gradients
        std::uniform_real_distribution<double> angDist(0.0, 2.0 * M_PI);
        vector<vector<pair<double,double>>> gradients(res.first+1, vector<pair<double,double>>(res.second+1));
        for (int i=0;i<=res.first;++i) for (int j=0;j<=res.second;++j) {
            double ang = angDist(grid_rng);
            gradients[i][j] = {cos(ang), sin(ang)};
        }
        if (tileable.first) for (int j=0;j<=res.second;++j) gradients[res.first][j] = gradients[0][j];
        if (tileable.second) for (int i=0;i<=res.first;++i) gradients[i][res.second] = gradients[i][0];

        // expand gradients to pixel resolution by repeating blocks
        auto gradAt = [&](int i, int j)->pair<double,double> {
            // which cell in gradient grid corresponds?
            int gi = min(i / d0, res.first); // careful with edge
            int gj = min(j / d1, res.second);
            return gradients[gi][gj];
        };

        // compute g00,g10,g01,g11 and ramps
        for (int i=0;i<H;++i) {
            for (int j=0;j<W;++j) {
                // grid fractional coords
                double gx = grid[i][j].first;
                double gy = grid[i][j].second;

                // indices for gradients (top-left corner of cell)
                int cell_i = (int)floor((double)i / d0);
                int cell_j = (int)floor((double)j / d1);
                // clamp to grid
                cell_i = std::min(cell_i, res.first);
                cell_j = std::min(cell_j, res.second);

                // get gradients for corners
                auto g00 = gradients[cell_i    ][cell_j    ];
                auto g10 = gradients[min(cell_i+1,res.first)][cell_j    ];
                auto g01 = gradients[cell_i    ][min(cell_j+1,res.second)];
                auto g11 = gradients[min(cell_i+1,res.first)][min(cell_j+1,res.second)];

                // compute dot products
                double n00 = g00.first * gx       + g00.second * gy;
                double n10 = g10.first * (gx-1.0) + g10.second * gy;
                double n01 = g01.first * gx       + g01.second * (gy-1.0);
                double n11 = g11.first * (gx-1.0) + g11.second * (gy-1.0);

                // interpolation
                double tx = interpolant(gx);
                double ty = interpolant(gy);
                double n0 = n00*(1.0 - tx) + tx * n10;
                double n1 = n01*(1.0 - tx) + tx * n11;
                double val = sqrt(2.0) * ((1.0 - ty) * n0 + ty * n1);
                data[n][i][j] = val;
            }
        }
    }
    return data;
}

Noise3D generate_fractal_noise_2d(pair<int,int> shape, pair<int,int> res, int octaves=1, double persistence=0.5, double lacunarity=2.0, pair<bool,bool> tileable={false,false}, unsigned seed=0, int number=1) {
    Noise3D noise(number, vector<vector<double>>(shape.first, vector<double>(shape.second, 0.0)));
    // generate base perlin at base frequency, then add octaves
    for (int n=0;n<number;++n) {
        double frequency = 1.0;
        double amplitude = 1.0;
        // we can't re-use generate_perlin_noise_2d for differing frequencies easily,
        // so call it with appropriate res for each octave and add scaled results.
        for (int o=0;o<octaves;++o) {
            pair<int,int> r = { int(frequency * res.first), int(frequency * res.second) };
            auto base = generate_perlin_noise_2d(shape, r, tileable, seed + n + o*97, 1);
            for (int i=0;i<shape.first;++i) for (int j=0;j<shape.second;++j) {
                noise[n][i][j] += amplitude * base[0][i][j];
            }
            frequency *= lacunarity;
            amplitude *= persistence;
        }
    }
    return noise;
}

/* ---------------------------
   MemoryN classes (agents)
   --------------------------- */

const int COOP = 1;
const int DEF = 0;

struct MemoryN {
    int N;
    int maxN;
    vector<double> rule; // length 4^maxN
    unsigned long long startString;
    unsigned long long prev;
    double score;
    string name;

    MemoryN(int _N=0, unsigned long long startS=0, int _maxN=1) {
        N = _N;
        maxN = _maxN;
        startString = startS;
        prev = startString;
        size_t len = 1ULL << (2*maxN); // 4**maxN
        rule.assign(len, 0.0);
        score = 0.0;
        name = "MemoryN";
    }

    virtual ~MemoryN() = default;

    virtual void startup(unsigned long long startString) {
        this->startString = startString;
        this->prev = startString;
        this->score = 0.0;
    }

    void addToPast(int move) {
        // lsb is most recent, msb first move
        unsigned long long mask = (1ULL << maxN) - 1ULL;
        prev = ((prev << 1) + (unsigned long long)move) & mask;
    }

    int playMove(unsigned long long theirPast, double seed, int roundNum) {
        if (roundNum < N) {
            // return bit at position roundNum of startString
            unsigned long long mask = 1ULL << roundNum;
            return ( (startString & mask) ? COOP : DEF );
        }
        unsigned long long myPastMask = (prev & ((1ULL << N) - 1ULL));
        unsigned long long theirPastMask = (theirPast & ((1ULL << N) - 1ULL));
        unsigned long long ruleKey = (myPastMask << N) + theirPastMask;
        double prob = rule[ruleKey];
        if (seed < prob) {
            addToPast(COOP);
            return COOP;
        } else {
            addToPast(DEF);
            return DEF;
        }
    }

    void reset() {
        score = 0.0;
        prev = startString;
    }

    virtual string repr() const {
        return name;
    }
};

struct BLANK : public MemoryN {
    BLANK(int _N=0, unsigned long long startS=0, int _maxN=1) : MemoryN(_N, startS, _maxN) {
        name = "BLANK";
        size_t len = 1ULL << (2*maxN);
        rule.assign(len, 0.0);
        startString = startS;
        prev = startS;
    }
    void setRule(const vector<double> &r) {
        rule = r;
    }
};

struct TFT : public MemoryN {
    TFT(int _maxN=1) : MemoryN(1, 1, _maxN) {
        name = "TFT";
        N = 1;
        size_t len = 1ULL << (2*maxN);
        rule.assign(len, 0.0);
        // rule[1::2] = 1 in Python means all odd indices cooperate
        for (size_t i=0;i<rule.size();++i) if ( (i & 1) == 1 ) rule[i] = 1.0;
        startString = COOP;
        prev = COOP;
    }
};

struct DU : public MemoryN {
    DU(int _maxN=1) : MemoryN(0, DEF, _maxN) {
        name = "DU";
        N = 0;
        size_t len = 1ULL << (2*maxN);
        rule.assign(len, 0.0);
        startString = DEF;
        prev = DEF;
    }
};

struct CU : public MemoryN {
    CU(int _maxN=1) : MemoryN(0, COOP, _maxN) {
        name = "CU";
        N = 0;
        size_t len = 1ULL << (2*maxN);
        rule.assign(len, 1.0);
        startString = COOP;
        prev = COOP;
    }
};

struct RandAgent : public MemoryN {
    RandAgent(int _maxN=1) : MemoryN(0, 0, _maxN) {
        name = "RAND";
        N = 0;
        size_t len = 1ULL << (2*maxN);
        rule.assign(len, 0.5);
        startString = (uniform01() < 0.5 ? COOP : DEF);
        prev = startString;
    }
    string repr() const override { 
        return name + " " + to_string((int)startString);
    }
};

/* ---------------------------
   Grid generation
   --------------------------- */

using AgentGrid = vector<vector<shared_ptr<MemoryN>>>;

// randomGrid(N, maxN)
AgentGrid randomGrid(int N, int maxN=1) {
    AgentGrid grid(N, vector<shared_ptr<MemoryN>>(N));
    for (int r=0;r<N;++r) for (int c=0;c<N;++c) {
        grid[r][c] = make_shared<RandAgent>(maxN);
    }
    return grid;
}

// blankGrid(N, res, maxN=1, seed)
AgentGrid blankGrid(int N, pair<int,int> res, int maxN=1, unsigned seed = 0) {
    if (seed==0) seed = (unsigned) (uniform01() * 1000.0);
    cout << "seed: " << seed << "\n";
    // number parameter in original called number=(4**maxN) => use that many channels
    int number = 1;
    for (int i=0;i<maxN;++i) number *= 4;
    auto paramMaps = generate_fractal_noise_2d({N,N}, res, /*octaves=*/1, /*persistence=*/0.5, /*lacunarity=*/2.0, {false,false}, seed, number);
    // paramMaps is number x N x N
    // python code adds +1.3 then divides by 2
    for (int k=0;k<number;++k) for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        paramMaps[k][i][j] += 1.3;
        paramMaps[k][i][j] /= 2.0;
    }
    AgentGrid grid(N, vector<shared_ptr<MemoryN>>(N));
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        auto ag = make_shared<BLANK>(1, 1, maxN);
        // create rule vector of length 4**maxN. The python did: setRule([i**2 for i in list(paramMaps[:,idr,idc])])
        // paramMaps[:,idr,idc] is "number" values — they square them.
        vector<double> rule(1ULL << (2*maxN), 0.0);
        // If number (channels) does not equal rule size, we'll distribute or repeat
        // but the Python used list(paramMaps[:,idr,idc]) and squared those values to form rule,
        // meaning rule length == number. But rule length should be 4**maxN. In the python blankGrid,
        // they use number=(4**maxN) so number == rule length. So here it's consistent.
        if ((int)rule.size() != number) {
            // fallback: fill uniform
            for (size_t rr=0; rr<rule.size(); ++rr) rule[rr] = 0.5;
        } else {
            for (int k=0;k<number;++k) {
                double v = paramMaps[k][i][j];
                rule[k] = v * v;
                // clamp [0,1]
                if (rule[k] < 0.0) rule[k] = 0.0;
                if (rule[k] > 1.0) rule[k] = 1.0;
            }
        }
        ag->setRule(rule);
        grid[i][j] = ag;
    }
    return grid;
}

/* ---------------------------
   Tournaments
   --------------------------- */

static vector<vector<double>> payoffMatrix = {{1,5},{0,3.01}};

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

vector<vector<vector<double>>> agentRuleSnapshot(const AgentGrid &agents) {
    int Y = (int)agents.size();
    int X = (int)agents[0].size();
    int ruleLen = (int)agents[0][0]->rule.size();
    vector<vector<vector<double>>> snap(Y, vector<vector<double>>(X, vector<double>(ruleLen)));
    for (int i=0;i<Y;++i) for (int j=0;j<X;++j) {
        for (int k=0;k<ruleLen;++k) snap[i][j][k] = agents[i][j]->rule[k];
    }
    return snap;
}

TorusResult torusTournament(AgentGrid agentGrid, int iters, int rounds, int snaps, float evolutionRate, float evolutionChance, float mutationRate) {
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

        // We'll parallelize by rows (each row handles its matches)
        int nThreads = min((int)thread::hardware_concurrency(), yLen);
        vector<thread> threads;
        mutex scoreMutex; // not necessary since each thread writes distinct cells, but safe when updating opponents
        auto worker = [&](int startRow, int endRow) {
            // each row processes its matches
            for (int idy = startRow; idy < endRow; ++idy) {
                for (int idx = 0; idx < xLen; ++idx) {
                    auto match = matchups[idy][idx];
                    auto a1 = agentGrid[idy][idx];
                    auto a2 = agentGrid[match.second][match.first]; // note match stored as (x,y) earlier
                    // increment played counts (both)
                    playedTracker[idy][idx] += 1;
                    playedTracker[match.second][match.first] += 1;
                    // play iters rounds between a1 and a2
                    vector<double> seeds(2*iters);
                    for (int s=0;s<2*iters;++s) seeds[s] = uniform01();
                    for (int n=0;n<iters;++n) {
                        unsigned long long a1prev = a1->prev;
                        unsigned long long a2prev = a2->prev;
                        int a1move = a1->playMove(a2prev, seeds[n], n);
                        int a2move = a2->playMove(a1prev, seeds[n+iters], n);
                        scoreTracker[idy][idx] += payoffMatrix[a1move][a2move];
                        scoreTracker[match.second][match.first] += payoffMatrix[a2move][a1move];
                        // reset is done after loop
                    }
                    a1->reset();
                    a2->reset();
                }
            }
        };

        // launch threads
        int rowsPerThread = max(1, yLen / nThreads);
        int row = 0;
        for (int t=0;t<nThreads;++t) {
            int startR = row;
            int endR = min(yLen, row + rowsPerThread);
            if (t == nThreads-1) endR = yLen;
            threads.emplace_back(worker, startR, endR);
            row = endR;
        }
        for (auto &th : threads) if (th.joinable()) th.join();
        threads.clear();

        // Normalize by playedTracker (avoid div by zero)
        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) {
            if (playedTracker[i][j] > 0) scoreTracker[i][j] /= (double)playedTracker[i][j];
            totalScore[i][j] += scoreTracker[i][j];
        }

        // Evolution
        AgentGrid newGrid = agentGrid; // shallow copy of shared_ptrs
        double shiftPercentage = 0.2;
        double mutationRate = 0.01;
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
                    ruleShift2[k] = ((uniform01() * 2.0) - 1.0) * mutationRate;
                }
                vector<double> newRule = agentGrid[idy][idx]->rule;
                for (size_t k=0;k<newRule.size();++k) newRule[k] = newRule[k] + ruleShift[k] + ruleShift2[k];
                for (size_t k=0;k<newRule.size();++k) {
                    if (newRule[k] < 0.0) newRule[k] = 0.0;
                    if (newRule[k] > 1.0) newRule[k] = 1.0;
                }
                // assign to newGrid copy
                // make a fresh BLANK agent to hold new rule while preserving other meta
                auto newAgent = make_shared<BLANK>(agentGrid[idy][idx]->N, agentGrid[idy][idx]->startString, agentGrid[idy][idx]->maxN);
                newAgent->name = agentGrid[idy][idx]->name;
                newAgent->rule = newRule;
                newAgent->startString = agentGrid[idy][idx]->startString;
                newAgent->prev = agentGrid[idy][idx]->prev;
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
            int ruleLen = (int)agentGrid[0][0]->rule.size();
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

// torusTournamentNoEvolution is similar but without the evolution stage
TorusResult torusTournamentNoEvolution(AgentGrid agentGrid, int iters, int rounds, int snaps) {
    int yLen = (int)agentGrid.size();
    int xLen = (int)agentGrid[0].size();
    int N = yLen * xLen;
    TorusResult out;
    vector<vector<double>> totalScore(yLen, vector<double>(xLen, 0.0));
    int snapEvery = max(1, rounds / snaps);

    for (int round=0; round<rounds; ++round) {
        auto matchups = pickOpponents(agentGrid);
        vector<vector<int>> playedTracker(yLen, vector<int>(xLen, 0));
        vector<vector<double>> scoreTracker(yLen, vector<double>(xLen, 0.0));

        // parallelize by rows
        int nThreads = min((int)thread::hardware_concurrency(), yLen);
        vector<thread> threads;
        auto worker = [&](int startRow, int endRow) {
            for (int idy=startRow; idy<endRow; ++idy) {
                for (int idx=0; idx<xLen; ++idx) {
                    auto match = matchups[idy][idx];
                    auto a1 = agentGrid[idy][idx];
                    auto a2 = agentGrid[match.second][match.first];
                    playedTracker[idy][idx] += 1;
                    playedTracker[match.second][match.first] += 1;
                    vector<double> seeds(2*iters);
                    for (int s=0;s<2*iters;++s) seeds[s] = uniform01();
                    for (int n=0;n<iters;++n) {
                        unsigned long long a1prev = a1->prev;
                        unsigned long long a2prev = a2->prev;
                        int a1move = a1->playMove(a2prev, seeds[n], n);
                        int a2move = a2->playMove(a1prev, seeds[n+iters], n);
                        scoreTracker[idy][idx] += payoffMatrix[a1move][a2move];
                        scoreTracker[match.second][match.first] += payoffMatrix[a2move][a1move];
                    }
                    a1->reset(); a2->reset();
                }
            }
        };
        int rowsPerThread = max(1, yLen / nThreads);
        int row = 0;
        for (int t=0;t<nThreads;++t) {
            int s = row;
            int e = min(yLen, row + rowsPerThread);
            if (t == nThreads-1) e = yLen;
            threads.emplace_back(worker, s, e);
            row = e;
        }
        for (auto &th : threads) if (th.joinable()) th.join();
        threads.clear();

        for (int i=0;i<yLen;++i) for (int j=0;j<xLen;++j) {
            if (playedTracker[i][j] > 0) scoreTracker[i][j] /= (double)playedTracker[i][j];
            totalScore[i][j] += scoreTracker[i][j];
        }

        if (round == 1 || ((round % snapEvery) == 0 && (round / snapEvery) > 0)) {
            out.scoreSnaps.push_back(totalScore);
            auto ruleSnap = agentRuleSnapshot(agentGrid);
            int ruleLen = (int)agentGrid[0][0]->rule.size();
            vector<vector<double>> flatRules(yLen, vector<double>(xLen * ruleLen));
            for (int iy=0; iy<yLen; ++iy) for (int ix=0; ix<xLen; ++ix)
                for (int k=0;k<ruleLen;++k) flatRules[iy][ix*ruleLen + k] = ruleSnap[iy][ix][k];
            out.ruleSnaps.push_back(flatRules);
            out.nonCumulativeScoreSnaps.push_back(scoreTracker);
            cout << "progress: " << (round / snapEvery) << " / " << snaps << "\n";
        }
    }

    out.totalScore = totalScore;
    return out;
}

/* ---------------------------
   Output helpers (CSV)
   --------------------------- */

void write_scoreSnaps_csv(const vector<vector<vector<double>>> &scoreSnaps, const string &fname) {
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
    ofstream f(fname);
    // Each line: snap_index,y,x,ruleIndex,ruleValue   (sparse long form)
    for (size_t s=0; s<ruleSnaps.size(); ++s) {
        auto &flatRules = ruleSnaps[s]; // y -> x*ruleLen
        int y = (int)flatRules.size();
        int Xflat = (int)flatRules[0].size();
        // we don't know ruleLen directly; but it's Xflat / xLen. To keep things simple, output flattened full lines:
        // for each row write all values as a long comma-separated line (snap per line)
        for (int i=0;i<y;++i) {
            for (int j=0;j<Xflat;++j) {
                f << flatRules[i][j];
                if (j < Xflat-1) f << ",";
            }
            f << "\n";
        }
    }
    f.close();
}

void write_nonCumulative_csv(const vector<vector<vector<double>>> &ncs, const string &fname) {
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
    if (argc < 6) {
        cerr << "Usage: ./sim p00 p01 p10 p11 gridN res0 res1 maxN rounds iters snaps evolutionRate mutationRate evolutionChance seed gridseed\n";
        return 1;
    }

    payoffMatrix = {
        {atof(argv[1]), atof(argv[2])},
        {atof(argv[3]), atof(argv[4])}
    };

    cout << "Payoff matrix:" << endl;
    for (auto &row : payoffMatrix) {
        for (auto v : row) cout << v << " ";
        cout << endl;
    }
    int gridN = atoi(argv[5]);
    pair<int,int> res = {atoi(argv[6]),atoi(argv[7])};
    int maxN = atoi(argv[8]);
    int rounds = atoi(argv[9]);
    int iters = atoi(argv[10]);
    int snaps = atoi(argv[11]);
    double evolutionRate = atof(argv[12]);
    double mutationRate = atof(argv[13]);
    double evolutionChance = atof(argv[14]);
    unsigned int seed = (unsigned) std::atoi(argv[15]);
    global_rng.seed(seed);
    unsigned int gridseed = (unsigned) std::atoi(argv[16]);
    global_rng.seed(gridseed);

    std::ofstream file("params.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    cout << "Building grid...\n";
    AgentGrid grid = blankGrid(gridN, res, maxN, gridseed);

    // Write CSV header
    file << "Parameter,Value\n";

    // Write each parameter
    file << "gridN," << gridN << "\n";
    file << "res_x," << res.first << "\n";
    file << "res_y," << res.second << "\n";
    file << "maxN," << maxN << "\n";
    file << "rounds," << rounds << "\n";
    file << "iters," << iters << "\n";
    file << "snaps," << snaps << "\n";
    file << "evolutionRate," << evolutionRate << "\n";
    file << "mutationRate," << mutationRate << "\n";
    file << "evolutionChance," << evolutionChance << "\n";
    file << "p00," << payoffMatrix[0][0] <<"\np01,"<< payoffMatrix[0][1] << "\np10," << payoffMatrix[1][0] << "\np11," << payoffMatrix[1][1] << "\n";

    file.close();
    std::cout << "Parameters written to params.csv successfully.\n";

    cout << "Running tournament (" << rounds << " rounds, " << iters << " iters per match)...\n";
    //TorusResult resu = torusTournamentNoEvolution(grid, iters, rounds, snaps);
    TorusResult resu = torusTournament(grid, iters, rounds, snaps, evolutionRate, mutationRate, evolutionChance);

    cout << "Writing CSV outputs...\n";
    write_scoreSnaps_csv(resu.scoreSnaps, "scoreSnaps.csv");
    write_totalScore_csv(resu.totalScore, "totalScore.csv");
    write_ruleSnaps_csv(resu.ruleSnaps, "ruleSnaps.csv");
    write_nonCumulative_csv(resu.nonCumulativeScoreSnaps, "nonCumulativeScore.csv");

    cout << "Done. Files created: scoreSnaps.csv, totalScore.csv, ruleSnaps.csv, nonCumulativeScore.csv\n";
    return 0;
}


int mainDeprecated(int argc, char** argv) {
    // Parameters mirror your testing.py
    int gridN = 32;
    pair<int,int> res = {2,2};
    int maxN = 1;
    int rounds = 10000;
    int iters = 100;
    int snaps = 500;
    double evolutionRate = 0.1;
    double mutationRate = 0.001;
    double evolutionChance = 0.2;

    payoffMatrix = {{1,5},{0,3.2}};

    std::ofstream file("params.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    // Write CSV header
    file << "Parameter,Value\n";

    // Write each parameter
    file << "gridN," << gridN << "\n";
    file << "res_x," << res.first << "\n";
    file << "res_y," << res.second << "\n";
    file << "maxN," << maxN << "\n";
    file << "rounds," << rounds << "\n";
    file << "iters," << iters << "\n";
    file << "snaps," << snaps << "\n";
    file << "evolutionRate," << evolutionRate << "\n";
    file << "mutationRate," << mutationRate << "\n";
    file << "evolutionChance," << evolutionChance << "\n";
    file << "p00," << payoffMatrix[0][0] <<"\np01,"<< payoffMatrix[0][1] << "\np10," << payoffMatrix[1][0] << "\np11," << payoffMatrix[1][1] << "\n";

    file.close();
    std::cout << "Parameters written to params.csv successfully.\n";

    cout << "Building grid...\n";
    AgentGrid grid = blankGrid(gridN, res, maxN, /*seed=*/0);

    cout << "Running tournament (" << rounds << " rounds, " << iters << " iters per match)...\n";
    //TorusResult resu = torusTournamentNoEvolution(grid, iters, rounds, snaps);
    TorusResult resu = torusTournament(grid, iters, rounds, snaps, evolutionRate, mutationRate, evolutionChance);

    cout << "Writing CSV outputs...\n";
    write_scoreSnaps_csv(resu.scoreSnaps, "scoreSnaps.csv");
    write_totalScore_csv(resu.totalScore, "totalScore.csv");
    write_ruleSnaps_csv(resu.ruleSnaps, "ruleSnaps.csv");
    write_nonCumulative_csv(resu.nonCumulativeScoreSnaps, "nonCumulativeScore.csv");

    cout << "Done. Files created: scoreSnaps.csv, totalScore.csv, ruleSnaps.csv, nonCumulativeScore.csv\n";
    return 0;
}

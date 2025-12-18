#define NOMINMAX
#include <fstream>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <sstream>

#include "network.hpp"
#include "torusNetwork.hpp"
#include "torusNetworkVonNeumann.hpp"

using namespace std;

// RNG
std::mt19937_64 global_rng;

double uniform01() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(global_rng);
}

int randint(int a, int b) {
    return a + (uniform01() * b);
}

enum Move: int {
    COOP = 1,
    DEF = 0
};

struct Memory1 {
    Move startMove;
    Move prevMove;
    array<double, 4> rule;
    array<double, 4> ruleTemp;
    double score = 0.0;
    double mutationRate = 0.01;

    Memory1(Move start = COOP, double mr=0.01):
        startMove(start),
        prevMove(startMove),
        rule{0.0,0.0,0.0,0.0},
        ruleTemp{0.0,0.0,0.0,0.0},
        mutationRate(mr),
        score(0.0)
    {}

    virtual void startup(Move start) {
        this->startMove = start;
        this->prevMove = start;
        this->score = 0.0;
    }

    Move playMove(Move theirPrev, double seed, int roundNum) {
        double prob = 0;
        if ((theirPrev == DEF) and (prevMove == DEF)) {
            prob = rule[0];
        } else if ((theirPrev == DEF) and (prevMove == COOP)) {
            prob = rule[1];
        } else if ((theirPrev == COOP) and (prevMove == DEF)) {
            prob = rule[2];
        } else {
            prob = rule[3];
        }

        if (seed < prob) {
            prevMove = COOP;
            return COOP;
        } else {
            prevMove = DEF;
            return DEF;
        }
    }

    void addScore(int toAdd) {
        score += toAdd;
    }

    void reset() {
        score = 0.0;
        prevMove = startMove;
    }

    void setRule(const array<double, 4> &r) {
        rule[0] = r[0];
        rule[1] = r[1];
        rule[2] = r[2];
        rule[3] = r[3];
        setRuleTemp(r);
    }

    void setRuleTemp(const array<double, 4> &r) {
        ruleTemp[0] = r[0];
        ruleTemp[1] = r[1];
        ruleTemp[2] = r[2];
        ruleTemp[3] = r[3];
    }
};

vector<vector<vector<double>>>
read_blocks_csv(const string &filename, int N) {

    vector<vector<vector<double>>> data(
        5, vector<vector<double>>(N, vector<double>(N))
    );

    ifstream file(filename);
    if (!file) {
        throw runtime_error("Error opening file: " + filename);
    }

    std::string line;
    int block = 0;
    int row_in_block = 0;

    while (getline(file, line)) {
        // skip empty lines if they appear
        if (line.size() == 0) continue;

        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (getline(ss, cell, ',')) {
            data[block][row_in_block][col] = stod(cell);
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

unique_ptr<TorusGridNetworkVonNeumann<Memory1>>
importGrid(int N, const string& filename) {
    auto net = make_unique<TorusGridNetworkVonNeumann<Memory1>>(N, N);
    auto maps = read_blocks_csv(filename, N);

    std::cout << "Shape: (" 
        << maps.size() << ", "
        << (maps.empty() ? 0 : maps[0].size()) << ", "
        << (maps.empty() || maps[0].empty() ? 0 : maps[0][0].size()) << ")\n";

    for (int y=0; y<N; ++y) for (int x=0; x<N; ++x) {
        array<double,4> rule;
        for (int k=0;k<4;++k) rule[k] = clamp(maps[k][y][x], 0.0, 1.0);
        auto ag = make_shared<Memory1>(COOP, maps[4][y][x]);
        ag->setRule(rule);
        net->at(x,y) = ag;
    }
    return net;
}

vector<vector<pair<int,int>>> pickOpponents(const Network<Memory1>& net) {
    int H = net.height();
    int W = net.width();
    vector<vector<pair<int,int>>> opponents(H, vector<pair<int,int>>(W));
    for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
        auto neighbors = net.neighbors(x,y);
        opponents[y][x] = neighbors[randint(0,neighbors.size()-1)];
    }
    return opponents;
}

vector<vector<double>> agentRuleSnapshot(const Network<Memory1> &net) {
    int H = net.height();
    int W = net.width();

    vector<vector<double>> snap(H, vector<double>(4*W));
    vector<double> row(4*W);
    for (int y=0;y<H;++y) {
        for (int x=0;x<W;++x) {
            for (int k=0;k<4;++k) row[(4*x)+k] = net.at(x,y)->rule[k];
        }
        snap[y] = row;
    }
    return snap;
}

static vector<vector<double>> payoffMatrix = {{1,5},{0,3}};

struct TournamentResultTorus {
    vector<vector<vector<double>>> scoreSnaps;
    vector<vector<vector<double>>> ruleSnaps;
    vector<vector<vector<double>>> nonCumulativeScoreSnaps;
    vector<vector<double>> totalScore;
};

TournamentResultTorus tournament(
Network<Memory1> &net, 
int iters, 
int rounds,
int snaps, 
double evolutionRate,
double evoChance) {
    int H = net.height();
    int W = net.width();
    int N = H*W;

    TournamentResultTorus out;
    vector<vector<double>> totalScore(H, vector<double>(W));
    int snapEvery = max(1, rounds/snaps);

    vector<vector<int>> playedTracker(H, vector<int>(W, 0));
    vector<vector<double>> scoreTracker(H, vector<double>(W, 0.0));

    vector<vector<pair<int,int>>> matchups;

    for (int round = 0; round < rounds; ++round) {
        matchups = pickOpponents(net);

        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            pair<int,int> match = matchups[y][x];
            auto a1 = net.at(x,y);
            auto a2 = net.at(match.first, match.second);
            
            playedTracker[y][x] += 1;
            playedTracker[match.second][match.first] += 1;

            vector<double> seeds(2*iters);
            for (int s=0; s<2*iters; ++s) seeds[s] = uniform01();
            for (int n=0; n<iters; ++n) {
                Move a1prev = a1->prevMove;
                Move a2prev = a2->prevMove;
                Move a1move = a1->playMove(a2prev, seeds[n], n);
                Move a2move = a2->playMove(a1prev, seeds[n+1], n);
                scoreTracker[y][x] += payoffMatrix[a1move][a2move];
                scoreTracker[match.second][match.first] += payoffMatrix[a2move][a1move];
            }
            a1->reset();
            a2->reset();
        }

        for (int y=0;y<H;++y) for (int x=0;x<W;x++) {
            if (playedTracker[y][x] > 0) scoreTracker[y][x] /= (double)playedTracker[y][x];
            totalScore[y][x] += scoreTracker[y][x];
        }

        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            array<double,4> newRule = net.at(x,y)->rule;
            if (uniform01() > evoChance) {
                // Mutate
                for (int k=0;k<4;k++) {
                    newRule[k] += ((uniform01()*2)-1)*net.at(x,y)->mutationRate;
                    newRule[k] = clamp(newRule[k],0.0,1.0);
                }
                net.at(x,y)->setRule(newRule);
            } else {
                // Evolve
                auto neighbors = net.neighbors(x,y);
                double highScore = scoreTracker[y][x];
                int highX = x;
                int highY = y;
                for (int k=0;k<neighbors.size();k++) {
                    auto [nx, ny] = neighbors[k];
                    if (scoreTracker[ny][nx] > highScore) {
                        highScore = scoreTracker[ny][nx];
                        highX = nx;
                        highY = ny;
                    }
                }
                array<double, 4> ruleShift;
                array<double,4> targetRule = net.at(highX, highY)->rule;
                for (int k=0;k<4;k++) {
                    // Mutate
                    newRule[k] += ((uniform01()*2)-1)*net.at(x,y)->mutationRate;
                    newRule[k] += evolutionRate*(targetRule[k] - net.at(x,y)->rule[k]);
                    newRule[k] = clamp(newRule[k],0.0,1.0);
                }
                net.at(x,y)->setRuleTemp(newRule);
            }
        }

        for (int y=0;y<H;++y) for (int x=0;x<W;++x) 
            net.at(x,y)->setRule(net.at(x,y)->ruleTemp);

        if (round ==  0 || ((round % snapEvery) == 0 && (round / snapEvery) > 0)) {
            out.scoreSnaps.push_back(totalScore);
            out.ruleSnaps.push_back(agentRuleSnapshot(net));
            out.nonCumulativeScoreSnaps.push_back(scoreTracker);
            cout << "progress: " << (round / snapEvery) << " / "<<snaps<<"\n";
        }

        for (auto &row : playedTracker) {
            fill(row.begin(), row.end(), 0);
        }
        for (auto &row : scoreTracker) {
            fill(row.begin(), row.end(), 0);
        }
    }
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
    unique_ptr<Network<Memory1>> net = importGrid(gridN, path+"/maps.csv");

    cout << "Running tournament (" << rounds << " rounds, " << iters << " iters per match)...\n";
    TournamentResultTorus resu = tournament(*net, iters, rounds, snaps, evolutionRate, evolutionChance);

    write_scoreSnaps_csv(resu.scoreSnaps, path+"/scoreSnaps.csv");
    write_totalScore_csv(resu.totalScore, path+"/totalScore.csv");
    write_ruleSnaps_csv(resu.ruleSnaps, path+"/ruleSnaps.csv");
    write_nonCumulative_csv(resu.nonCumulativeScoreSnaps, path+"/nonCumulativeScore.csv");

    return 0;
}
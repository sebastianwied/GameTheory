#define NOMINMAX
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "splitmix.hpp"

using namespace std;

enum Move : int {
    DEF  = 0,
    COOP = 1
};

// Normalize rng output to [0, 1)
inline double rand_double(splitmix64& rng) {
    return (rng() >> 11) * (1.0 / (1ull << 53));
}

struct Players {
    vector<double> rules;
    vector<double> firstMoveProb;
    vector<Move>   prevMoves;
    vector<double> scores;

    array<double, 4> get_rule(int i) const {
        return {rules[4*i], rules[4*i+1], rules[4*i+2], rules[4*i+3]};
    }

    void set_rule(int i, const array<double, 4>& rule) {
        rules[4*i]   = rule[0];
        rules[4*i+1] = rule[1];
        rules[4*i+2] = rule[2];
        rules[4*i+3] = rule[3];
    }

    double get_score(int i) const { return scores[i]; }
    void add_score(int i, double add) { scores[i] += add; }
    void set_score(int i, double score) { scores[i] = score; }
    void scale_score(int i, double scale_factor) { scores[i] = scores[i] * scale_factor; }

    Move get_prev_move(int i) const { return prevMoves[i]; }
    void set_prev_move(int i, Move move) { prevMoves[i] = move; }

    Move get_first_move(int i, splitmix64& rng) const {
        return rand_double(rng) < firstMoveProb[i] ? COOP : DEF;
    }

    void set_first_move(int i, double prob) { firstMoveProb[i] = prob; }

    // Rule index: (2 * their_prev) + my_prev
    // [0]=DD, [1]=CD, [2]=DC, [3]=CC  (my prev, their prev)
    Move get_move(int i, Move my_prev, Move their_prev, splitmix64& rng) {
        double prob = get_rule(i)[(2 * their_prev) + my_prev];
        return rand_double(rng) < prob ? COOP : DEF;
    }
};

struct Graph {
    vector<int> neighbors;
    vector<int> offsets;
    int N;

    int sample_neighbor(int i, splitmix64& rng) const {
        int start = offsets[i];
        int deg   = offsets[i+1] - start;
        return neighbors[start + rng() % deg];
    }

    int best_neighbor(int i, const Players& players) const {
        int start = offsets[i];
        int end   = offsets[i+1];
        int best  = i;
        for (int k = start; k < end; ++k) {
            if (players.get_score(neighbors[k]) > players.get_score(best))
                best = neighbors[k];
        }
        return best;
    }
};

Graph load_graph(const string& graph_file, const string& rule_file, Players& players) {
    Graph g;
    int N = 0;
    ifstream gfile(graph_file);
    ifstream rfile(rule_file);
    string gline, rline;

    while (getline(gfile, gline) && getline(rfile, rline)) {
        g.offsets.push_back(g.neighbors.size());
        istringstream gss(gline);
        string token;
        while (getline(gss, token, ','))
            g.neighbors.push_back(stoi(token));

        istringstream rss(rline);
        for (int k = 0; k < 4; ++k) {
            getline(rss, token, ',');
            players.rules.push_back(stod(token));
        }
        ++N;
    }
    g.offsets.push_back(g.neighbors.size());
    g.N = N;

    players.scores.assign(N, 0.0);
    players.prevMoves.assign(N, DEF);
    players.firstMoveProb.assign(N, 0.5);

    return g;
}

void take_snapshot(int round, const Players& players, ofstream& out, double& cooperation_frac) {
    double r = round;
    out.write(reinterpret_cast<const char*>(&r), sizeof(double));
    out.write(reinterpret_cast<const char*>(&cooperation_frac), sizeof(double));
    out.write(reinterpret_cast<const char*>(players.scores.data()),
              players.scores.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(players.rules.data()),
              players.rules.size() * sizeof(double));
}

struct Game {
    double evolution_chance;
    double evolution_rate;
    double mutation_rate;
    int rounds;
    int iterations;
    // Payoff indexed as (2*p2move + p1move)
    // [0]=P(DD), [1]=S(CD), [2]=T(DC), [3]=R(CC)
    array<double, 4> payoff;

    double get_score(Move p1, Move p2) const {
        return payoff[(2 * p2) + p1];
    }

    void play_first_move(Players& players, int p1, int p2, splitmix64& rng, int& cooperation_count) {
        Move p1move = players.get_first_move(p1, rng);
        Move p2move = players.get_first_move(p2, rng);
        cooperation_count += p1move + p2move;
        players.set_prev_move(p1, p1move);
        players.set_prev_move(p2, p2move);
        players.add_score(p1, get_score(p1move, p2move));
        players.add_score(p2, get_score(p2move, p1move));
    }

    void play_move(Players& players, int p1, int p2, splitmix64& rng, int& cooperation_count) {
        Move p1prev = players.get_prev_move(p1);
        Move p2prev = players.get_prev_move(p2);
        Move p1move = players.get_move(p1, p1prev, p2prev, rng);
        Move p2move = players.get_move(p2, p2prev, p1prev, rng);
        cooperation_count += p1move + p2move;
        players.set_prev_move(p1, p1move);
        players.set_prev_move(p2, p2move);
        players.add_score(p1, get_score(p1move, p2move));
        players.add_score(p2, get_score(p2move, p1move));
    }

    void evolve(Players& players, Graph& g, splitmix64& rng) {
        for (int i = 0; i < g.N; ++i) {
            if (rand_double(rng) < evolution_chance) {
                array<double, 4> new_rule = players.get_rule(g.best_neighbor(i, players));
                array<double, 4> my_rule = players.get_rule(i);
                array<double, 4> rule_shift = {0,0,0,0};
                for (int k=0; k < 4; ++k) {
                    rule_shift[k] = (new_rule[k] - my_rule[k]) * evolution_rate;
                    rule_shift[k] += mutation_rate*2*(rand_double(rng)-0.5);
                    new_rule[k] = clamp(my_rule[k] + rule_shift[k], 0.0, 1.0);
                }
                players.set_rule(i, new_rule);
            }
        }
    }

    void sim_loop(Graph& g, Players& players, splitmix64& rng,
                ofstream& out, int snapshot_interval) {
        int cooperation_count;
        int games;
        double cooperation_frac;
        vector<int> games_played;
        for (int i=0; i<g.N; ++i) {
            games_played.push_back(0);
        }
        for (int round = 0; round < rounds; ++round) {
            cooperation_count = 0;
            cooperation_frac = 0;
            games = 0;
            for (int p = 0; p < g.N; ++p) {players.set_score(p, 0); }
            for (int p1 = 0; p1 < g.N; ++p1) {
                games_played[p1] += 1;
                int p2 = g.sample_neighbor(p1, rng);
                games_played[p2] += 1;
                play_first_move(players, p1, p2, rng, cooperation_count);
                games += 2;
                for (int iter = 0; iter < iterations - 1; ++iter)
                    play_move(players, p1, p2, rng, cooperation_count);
                    games += 2;
            }
            for (int i=0; i<g.N; ++i) {
                players.scale_score(i, 1 / games_played[i]); 
                games_played[i] = 0;
            }
            evolve(players, g, rng);
            cooperation_frac = games / cooperation_count;
            if (round % snapshot_interval == 0)
                take_snapshot(round, players, out, cooperation_frac);
        }
        take_snapshot(rounds, players, out, cooperation_frac); // final snapshot
    }
};

int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: sim <graph_csv> <rules_csv> <output_bin> <rounds> <iters> <seed> <snaps> <evolution_chance> <evolution_rate> <mutation_rate>\n";
        return 1;
    }

    string graph_file  = argv[1];
    string rules_file  = argv[2];
    string output_file = argv[3];
    int rounds         = stoi(argv[4]);
    int iters          = stoi(argv[5]);
    int seed           = stoi(argv[6]);
    int snaps          = stoi(argv[7]);
    Game game;
    game.rounds     = rounds;
    game.iterations = iters;
    game.evolution_chance = stod(argv[8]);
    game.evolution_rate = stod(argv[9]);;
    game.mutation_rate = stod(argv[10]);

    Players players;
    Graph g = load_graph(graph_file, rules_file, players);

    game.payoff = {1.0, 0.0, 5.0, 3.0};  // P, S, T, R

    splitmix64 rng(seed);
    ofstream out(output_file, ios::binary);

    int snapshot_interval = max(1, rounds / snaps);
    game.sim_loop(g, players, rng, out, snapshot_interval);

    cout << "Done. Wrote snapshots to " << output_file << "\n";
    return 0;
}

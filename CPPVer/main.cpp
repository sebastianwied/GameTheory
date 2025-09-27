#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <numeric> // For std::iota
#include <fstream>   // For file output
using namespace std;
#define DEF 0
#define COOP 1
#define T 5
#define R 3
#define P 1
#define S 0

class Agent {
    public:
        Agent();
        string agType;
        string repr() { return "Score: " + to_string(points); }
        int prev;

    friend class Arbiter;

    friend std::vector<std::vector<int>> playNIterations(int n, Agent* a1, Agent* a2) {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<std::vector<int>> scores(2, std::vector<int>(n));
        // round one:
        int move = a1->turnOne + (2*(a2->turnOne));
        auto [s1, s2] = distributeScore(a1, a2, move);
        scores[0][0] = s1;
        scores[1][0] = s2;
        // rest of the rounds:
        for (int i = 1; i <= n-1; ++i) {
            move = getMove(dis(gen), dis(gen), a1, a2);
            auto [s1, s2] = distributeScore(a1, a2, move);
            scores[0][i] = scores[0][i-1] + s1;
            scores[1][i] = scores[1][i-1] + s2;
        }
        return scores;
    }

    friend int getMove(float seed1, float seed2, Agent* ag1, Agent* ag2) {
        int ag1move = ag1->playMove(seed1, ag2->prev);
        int ag2move = ag2->playMove(seed2, ag1->prev);
        int move = ag1move + (2*ag2move);
        // move key:
        // a1 controls whether +1(coop) or +0(def)
        // a2 controls whether +2(coop) or +0(def)
        // a1 def, a2 def = 0
        // a1 coop, a2 def = 1
        // a1 def, a2 coop = 2
        // a1 coop, a2 coop = 3
        return move;
    }

    friend std::pair<int,int> distributeScore(Agent* ag1, Agent* ag2, int move) {
        switch (move) {
            case 0:
                ag1->addPoints(P);
                ag2->addPoints(P);
                return std::make_pair(P, P);
            case 1:
                ag1->addPoints(S);
                ag2->addPoints(T);
                return std::make_pair(S, T);
            case 2:
                ag1->addPoints(T);
                ag2->addPoints(S);
                return std::make_pair(T, S);
            case 3:
                ag1->addPoints(R);
                ag2->addPoints(R);
                return std::make_pair(R, R);    
        }
        return std::make_pair(0, 0);
    }

    vector<float> rule; // memory-one strategy
    int points = 0; // score for this round(round != iteration!)
    int turnOne; // first played move, some strategies have a special start

    Agent(int type) {
        switch (type) {
            case 0:
                rule = std::vector<float>{1.0, 1.0, 1.0, 1.0};
                agType = "CU";
                turnOne = COOP;
                prev = COOP;
                break;
            case 1:
                rule = std::vector<float>{0.0, 0.0, 0.0, 0.0};
                agType = "DU";
                turnOne = DEF;
                prev = DEF;
                break;
            case 2:
                rule = std::vector<float>{0.0, 1.0, 0.0, 1.0};
                agType = "TFT";
                turnOne = COOP;
                prev = COOP;
                break;
            default:
                rule = std::vector<float>{0.5, 0.5, 0.5, 0.5};
                agType = "RAND";
                turnOne = COOP;
                prev = COOP;
                break;
        }
    }
    void setPoints(int newPoints) {
        points = newPoints;
    }
    void addPoints(int newPoints) {
        points += newPoints;
        return;
    }
    int playMove(float seed, int themPrev) {
        float prob = rule[(prev*2)+themPrev];
        //cout << "Prob: " << to_string(prob) << endl;
        if (seed < prob) {
            prev = COOP;
            return COOP; 
        }
        prev = DEF;
        return DEF;
    }
};

int main() {
    Agent ag1 = Agent(0);
    Agent ag2 = Agent(10);
    std::vector<std::vector<int>> scores = playNIterations(500, &ag1, &ag2);
    cout << ag1.repr() << "   " << ag2.repr() << endl;

    // Export data to CSV for Python plotting
    std::ofstream csvFile("game_results.csv");
    csvFile << "iteration,agent1_score,agent2_score\n";
    for (size_t i = 0; i < scores[0].size(); ++i) {
        csvFile << i+1 << "," << scores[0][i] << "," << scores[1][i] << "\n";
    }
    csvFile.close();
    return 0;
}

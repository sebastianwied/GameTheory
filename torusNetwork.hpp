#pragma once
#include "network.hpp"
#include <vector>

template <typename T>
class TorusGridNetwork: public Network<T> {
public:
    TorusGridNetwork(int h, int w):
    H(h), W(w),
    grid(h, std::vector<std::shared_ptr<T>>(w)) {}

    int height() const override {return H;}
    int width() const override {return W;}

    std::shared_ptr<T>& at(int x, int y) override {
        return grid[y][x];
    }

    const std::shared_ptr<T>& at(int x, int y) const override {
        return grid[y][x];
    }

    std::vector<std::pair<int, int>> neighbors(int x, int y) const override {
        int up = (y - 1 + H) % H;
        int left = (x - 1 + W) % W;
        int down = (y + 1) % H;
        int right = (x + 1) % W;
        return {
            {left, up}  , {x, up},   {right, up},
            {left, y}   ,            {right, y},
            {left, down}, {x, down}, {right, down},
        };
    }

private:
    int H, W;
    std::vector<std::vector<std::shared_ptr<T>>> grid;
};
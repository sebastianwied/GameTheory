#pragma once
#include <vector>
#include <memory>
#include <utility>

template<typename T>
class Network {
public:
    virtual ~Network() = default;

    virtual int height() const = 0;
    virtual int width() const = 0;

    virtual std::shared_ptr<T>& at(int x, int y) = 0;
    virtual const std::shared_ptr<T>& at(int x, int y) const = 0;

    // Return one opponent (or neighbor) for a given cell
    virtual std::vector<std::pair<int,int>> neighbors(int x, int y) const = 0;
};

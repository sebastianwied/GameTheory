CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall

sim: sim.cpp splitmix.hpp
	$(CXX) $(CXXFLAGS) -o sim sim.cpp

clean:
	rm -f sim

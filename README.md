# Docs

## File and program structure
### agents.py
This file holds code universal to agents and interactions between them. Add new strategies here.
Besides agents, it contains the code to:
- Define and distribute rewards according to the payoff matrix
- Play n iterations between agents. This is the backbone of the game.
Strategy structure - When creating new strategies, you need to include the following precisely:
- name string
- rule definition, taking only opponent move history(1d array of 0 or 1) and round number.
- function to return short version of name
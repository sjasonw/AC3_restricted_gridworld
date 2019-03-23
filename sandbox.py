import agent
import gridworld
import math
from gridworld import BoxGame
from ac3_grid import Agent

agent = Agent()
game = BoxGame(10)
agent.set_environment(game)
agent.episode_run()
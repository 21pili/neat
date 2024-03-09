import pygame
from pong import Game
import neat
import os
import pickle
import numpy as np
import graphviz
import warnings
import matplotlib.pyplot as plt



local_dir=os.path.dirname(__file__)
config_path=os.path.join(local_dir, "config.txt")

config=neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction
                            ,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
print(config)
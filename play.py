# import multiprocessing
import cProfile
import os
import pickle

from game import Game

import neat
# import neatpp.neat as neat


def play(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    with open('model_300', 'rb') as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = Game(network=winner_net, render=True)
    score = game.neatplay(0.6)
    # cProfile.run('game.neatplay(1)')
    print('Fitness:', score)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    # try:
    cProfile.run('play(config_path)', sort='tottime')
    # except ValueError:
        # pass

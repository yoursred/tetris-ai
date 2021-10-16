import os
import pickle

from game_tree import GameTree as Game

import neat


def play(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    with open('models/safe_50', 'rb') as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    game = Game(network=winner_net, render=True)
    score = game.tree_play(tickdelay=0.05, timeout=-1, movedelay=0.05)
    print('Fitness:', score)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-configs/feedforward-tree')

    play(config_path)

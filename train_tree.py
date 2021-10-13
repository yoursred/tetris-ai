import multiprocessing
import os
import pickle

import numpy as np
import neat

from game_tree import GameTree as Game


def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).
    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    for i in range(3):
        game = Game(network=net, render=False)
        fitnesses.append(game.tree_play(0.01, 10))

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        game = Game(network=net, render=True)
        genome.fitness = game.tree_play(0.01, 10)


def run(config_file, generations):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    # te = neat.ThreadedEvaluator(12, eval_genome)
    # winner = p.run(te.evaluate, generations)
    winner = p.run(pe.evaluate, generations)
    # winner = p.run(eval_genomes, generations)

    with open(f'models/safe_{generations}', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-configs/feedforward-tree')
    run(config_path, 50)

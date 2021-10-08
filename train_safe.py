import multiprocessing
import os
import pickle

# import numpy as np
import numpy as np

from game import Game

import neat
# import _neatpp.neat as neat
import cProfile



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

    # Run for up to 300 generations.
    # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    # te = neat.ThreadedEvaluator(12, eval_genome)
    # winner = p.run(te.evaluate, generations)
    # winner = p.run(pe.evaluate, generations)
    winner = p.run(eval_genomes, generations)

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = Game(network=winner_net)
    # score = game.neatplay()

    # print(f'Best genome\'s fitness: {score}')

    with open(f'model_{generations}', 'wb') as f:
        pickle.dump(winner, f)
    with open(f'pop_{generations}', 'wb') as f:
        pickle.dump(p, f)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    # try:
    run(config_path, 100)
    # cProfile.run('run(config_path, 10)', sort='tottime')
    # except ValueError:
    # pass

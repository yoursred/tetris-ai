from game import Game
import pickle

with open('/home/mcxreeper/999999', 'rb') as f:
    network = pickle.load(f)

game = Game(render=True, network=network)

game.tree_play(tickdelay=0.5, timeout=1000)
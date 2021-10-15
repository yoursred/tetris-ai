from functools import lru_cache
from copy import deepcopy
import time

import numpy as np

from game import Game
from constants import *


class GameTree(Game):
    """
    An agent that makes decisions more akin to Minimax. Machine learning is used here to determine the reward of a
    set of actions that is generated with the `GameTree.generate_legal_moves` method down bellow. Each moveset is
    applied on a copy of the playfield, the agent then picks the one with the highest reward. The reward is determined
    by feeding the network pre-defined properties of the playfield, they are:
        - Aggregate height
        - Number of holes
        - Number of columns with at least one hole
        - Bumpiness
        - Row transitions
        - Coloumn transitions
        - Number of empty columns
        - Deepest well
        - Lines cleared
        - Weighted blocks
    The network could be considered a playing policy.
    TODO implement search depth
    """

    def __init__(self, board: np.array = None, network=None, render=False):
        super(GameTree, self).__init__(board=board, network=network, render=render)

    @lru_cache(maxsize=64)
    def generate_legal_moves(self, c):
        """
        Generate legal (not useless) moves based on the provided piece, the piece has to be provided to allow easy
        caching as this function can take a toll on performance.
        :param c: The current piece
        :type c: Mino
        :return: List of movesets
        :rtype: list[tuple[str]]
        """
        moves = []

        for n, rot in enumerate(DIRECTIONS):
            test = c
            test.x, test.y = self.current.pos()
            test.direction = rot
            for h in (0, 1):
                for i in range(10 - max(test.render(), key=lambda x: x[0])[0]):
                    moves.append((
                        *('hold',) * h,
                        *('rotate',) * n,
                        *('right',) * i,
                        'drop'))
                for i in range(min(test.render(), key=lambda x: x[0])[0]):
                    moves.append((
                        *('hold',) * h,
                        *('rotate',) * n,
                        *('left',) * (i + 1),
                        'drop'))
        return moves

    def tree(self, d=0, c=None) -> dict:
        """
        Evaluate all current legal (not useless) movesets, and returns the one with the absolute highest reward. This
        means that this agents is deterministic.
        :return: Best moveset based on current situation
        :rtype: tuple[str]
        """

        # if c is None:
        #     c = copy(self)
        if c is None:
            legal = self.generate_legal_moves(self.current.copy())
        else:
            legal = self.generate_legal_moves(c.current.copy())
        tree = {
            move: [self.evaluate_move(move, c), 0]
            for move in legal
        }

        if d <= 0:
            return tree
        # print(tree)
        for k, v in tree.items():
            v[1] = self.tree(d - 1, v[0][1])

        return tree

    def evaluate_move(self, move, c=None):
        """
        Calculates the reward difference of the provided moveset.
        :param move: Set of moves
        :type move: tuple[str]
        :param c: Game state, will use current if None
        :return: Reward difference
        :rtype: float
        """
        if c is None:
            c = deepcopy(self)
            c.gameover = True

        for _ in move:
            c.game_step(_)
        return c.tree_fitness() - self.tree_fitness(), c

    def search_tree(self, tree, p=((), 0.0), fits=None):
        if fits is None:
            fits = set()

        for move, fc_t in tree.items():
            fits.add(
                (move + p[0], fc_t[0][0] + p[1])
            )
            if fc_t[1]:
                fits.update(self.search_tree(fc_t[1], (move, fc_t[0][0] + p[1]), fits))

        return fits

    def tree_play(self, tickdelay=0.01, timeout=6, movedelay=0.5, depth=0):
        """
        Play the game with the current network.
        :param tickdelay: How much time between asynchronous gravity ticks
        :type tickdelay: float
        :param timeout: How long before a gameover, if -1 no timeout
        :type timeout: float
        :param movedelay: How much time between move execution
        :type movedelay: float
        :param depth: How deep to make the decision tree TODO implement
        :type depth: int
        :return: Agent fitness after a gameover
        :rtype: float
        """
        depth += 0

        self.tickdelay = tickdelay
        self.time_left = timeout
        self.timetick()
        d = 0
        b = False
        if timeout == -1:
            b = True

        while (not self.gameover) and (self.time_left > 0 or b):
            if not b:
                d = time.time()

            tree = self.tree(0)

            fits = self.search_tree(tree)

            best = max(fits, key=lambda _: _[1])[0]

            # print(best)

            # for move in best:
            #     self.game_step(move)
            #     time.sleep(movedelay)

            if not b:
                d = time.time() - d
                self.time_left -= d
        return self.fitness

    def tree_fitness(self):
        """
        Calculate the absolute reward of the playfield
        :return:
        :rtype: float
        """
        cols = self.observations()[:, 1:]
        rows = cols.transpose()

        heights = []

        for col in cols:
            m = np.nonzero(col)[0]
            if m.size:
                heights.append(20 - min(m))
            else:
                heights.append(0)

        # Aggregate Height
        f0 = sum(heights)

        # Number of holes
        f1 = 0
        for col in cols:
            hit = False
            for square in col:
                if square and not hit:
                    hit = True
                if hit and not square:
                    f1 += 1

        # Number of columns with at least one hole
        f2 = 0
        for col in cols:
            hit = False
            for square in col:
                if square and not hit:
                    hit = True
                if hit and not square:
                    f2 += 1
                    break

        # Bumpiness
        f3 = 0
        for i, height in enumerate(heights[1:]):
            f3 += (height - heights[i - 1])

        # Row transitions
        f4 = 0
        for row in rows:
            current = row[0]
            for square in row:
                if current != square:
                    current = square
                    f4 += 1

        # Column transitions
        f5 = 0
        for col in cols:
            current = col[0]
            for square in col:
                if current != square:
                    current = square
                    f5 += 1

        # Empty columns
        f6 = sum([int(sum(x) == 0) for x in cols])

        wells = []
        for i in range(len(heights)):
            if i == 0:
                w = heights[1] - heights[0]
                w = w if w > 0 else 0
                wells.append(w)
            elif i == len(heights) - 1:
                w = heights[-2] - heights[-1]
                w = w if w > 0 else 0
                wells.append(w)
            else:
                w1 = heights[i - 1] - heights[i]
                w2 = heights[i + 1] - heights[i]
                w1 = w1 if w1 > 0 else 0
                w2 = w2 if w2 > 0 else 0
                w = w1 if w1 >= w2 else w2
                wells.append(w)

        # Deepest well
        f7 = max(wells)

        # Lines cleared
        f8 = self.last_cleared_lines

        # Weighted blocks
        f9 = 0
        for i, row in enumerate(rows):
            f9 += (20 - i) * sum(row)

        features = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

        return self.network.activate(features)[0]

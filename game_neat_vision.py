import time
from random import choices

import numpy as np

from game import Game


class GameNeatVision(Game):
    """
    This agent sucks ass, do not use it. That is all.
    """
    def __init__(self, board: np.array = None, network=None, render=False):
        super(GameNeatVision, self).__init__(board=board, network=network, render=render)

    def neatstep(self):
        activation = self.network.activate(
            self.observations().flatten()
        )

        choice = self._stepfromactivation(activation)
        self.current.pos_log.append((choice, tuple(self.current.pos()), self.current.direction))
        if self.render:
            self.paint(choice, activation)

    def neatplay(self, tickdelay=0.1, timeout=60):
        self.started_at = time.time()
        self.tickdelay = tickdelay
        self.timetick()
        while (not self.gameover) and time.time() - self.started_at < timeout:
            self.neatstep()
        return self.fitness

    def _stepfromactivation(self, activation):
        if not self.ticking:
            self.timetick()
        if sum(activation) == 0:
            activation = [1] * 6
        choice = choices(
            population=['left', 'right', 'rotate', 'drop', 'nop', 'hold'],
            weights=activation,
            k=1
        )[0]
        self.game_step(choice)
        return choice

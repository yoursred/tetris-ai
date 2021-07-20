# import pygame
import numpy as np
from numpy import pi, cos, sin, log as ln
# from numpy.random import choice as npchoice
from random import choices
from neat.nn.feed_forward import FeedForwardNetwork as FFN
from random import shuffle
import threading
from typing import Tuple
from copy import deepcopy



MINOS = {
    'I':((-2,-1),(-1,-1),(0,-1),(1,-1)),
    'J':((-1,0),(-1,-1),(0,0),(1,0)),
    'L':((-1,0),(1,-1),(0,0),(1,0)),
    'O':((-1,-1),(-1,0),(0,-1),(0,0)) ,# ((-1,1),(-1,0),(0,1),(0,0)),
    'S':((-1,0),(0,0),(0,-1),(1,-1)), # ((0,1),(1,1),(1,0),(2,0)),
    'T':((-1,0),(0,0),(0,-1),(1,0)),
    'Z':((-1,-1), (0,-1),(0,0),(1,0))
}

ROTATION_ORIGINS = { # What to rotate the tetrominos around
    'I':(-.5, -.5),
    'J':(0, 0),
    'L':(0, 0),
    'O':(-.5, -.5),
    'S':(0, 0),
    'T':(0, 0),
    'Z':(0, 0)
}

ANGLES = {
    'up':0,
    'right':pi/2,
    'down':pi,
    'left':3*pi/2
}

DIRECTIONS = {
    'up':'right',
    'right':'down',
    'down':'left',
    'left':'up'
}

WALL_KICKS = {
#     TODO: Implement wall kicks
}

SPAWN_POSITIONS = {
    'I' : (5, 20),
    'J' : (3, 21),
    'L' : (3, 21),
    'O' : (4, 21),
    'S' : (3, 21),
    'T' : (3, 21),
    'Z' : (3, 21)
}

INTS = {
    None: 0,
    'I' : 1,
    'J' : 2,
    'L' : 3,
    'O' : 4,
    'S' : 5,
    'T' : 6,
    'Z' : 7
}

# Rotation matrix
R = lambda theta: np.array([[cos(theta), -1*sin(theta)],[sin(theta), cos(theta)]])

# Rotate point around another
def rotate(point, theta, origin=(0,0)):
    offset = point - origin
    rotated = R(theta).dot(offset)
    return rotated + origin


# TODO: document
# TODO: Fix rotation validity check
# TODO: Turn off light before going to bed

class Mino:
    def __init__(self, type_: str, pos: Tuple[int, int]=None) -> None:
        if type_ not in 'IJLOSTZ':
            raise ValueError(f'invalid tetromino type {repr(type_)}')
        self.type = type_
        if pos is None:
            pos = SPAWN_POSITIONS[type_]
        self.x: int = pos[0]
        self.y: int = pos[1]
        self.direction = 'up'

    def render(self) -> list[tuple]:
        points = map(np.array, MINOS[self.type])
        points = map(
            rotate,
            points,
            (ANGLES[self.direction],) * 4,
            (ROTATION_ORIGINS[self.type],) * 4
        )
        points = map(
            lambda x: np.round(x).astype(int)+self.pos(), # numpy doesn't round when casting to int
                                                          # https://stackoverflow.com/a/43920513/16338589
            points
        )
        return list(points)

    def rotate(self) -> None:
        self._change_direction(DIRECTIONS[self.direction])

    def _change_direction(self, direction: str) -> None:
        if direction not in ('up', 'down', 'left', 'right'):
            raise ValueError(f'invalid direction: {repr(direction)}')
        self.direction = direction

    def copy(self):
        return Mino(self.type)

    def pos(self):
        return np.array((self.x, self.y))

    def check_collision(self, board, dirx=0, diry=1):
        for point in self.render():
            if point[0]+dirx not in range(10) or point[1]+diry not in range(40):
                return True
            if board[point[0]+dirx, point[1]+diry] != 0:
                return True
        return False

    def move(self, board, dir_):
        if dir_ == 'left':
            if not self.check_collision(board, -1, 0):
                self.x += -1
                # return False
            # self.move(board, 'down')
            return False
        elif dir_ == 'right':
            if not self.check_collision(board, 1, 0):
                self.x += 1
            # self.move(board, 'down')
            return False
        elif dir_ == 'rotate':
            test = Mino(self.type, (self.x, self.y))
            test._change_direction(self.direction)
            test.rotate()
            if not test.check_collision(board, 0, 0):
                self.rotate()
                return False
            # self.move(board, 'down')
            # return True
        elif dir_ == 'down': # or dir_ == 'nop':
            if not self.check_collision(board, 0, 1):
                self.y += 1
                return False
            return True
        if dir_ == 'drop':
            while not self.check_collision(board, 0, 1):
                self.y += 1
            return True
        if dir_ == 'nop':
            pass


class Game:
    def __init__(self, board: np.array=None, network: FFN=None):
        if board is None:
            self.board = np.zeros((10, 40))
        else:
            self.board = board
        self.network = network
        current, bag = self.generate_minos()
        self.current: Mino = current
        self.bag: list[Mino] = bag
        self.held = None
        self.score = 0
        self.gameover = False
        self.tickdelay = 1
        # self.gametimer.start()

    def step(self, cmd='nop'):
        if cmd not in ('nop', 'left', 'right', 'rotate', 'hold', 'drop', 'down'):
            raise ValueError(f'invalid command: {repr(cmd)}')
        if cmd != 'hold':
            if self.current.move(self.board, cmd):
                self.paste_to_board()
        elif cmd == 'hold':
            self.hold()


    @staticmethod
    def generate_minos():
        minos = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        # minos = ['I']
        shuffle(minos)
        for i, mino in enumerate(minos):
            minos[i] = Mino(mino)
        current, *minos = minos
        return current, minos

    def pop_from_bag(self, hold=False):
        if self.bag:
            if (self.held is None) or hold:
                self.current = self.bag.pop()
            else:
                self.current = self.held
                self.held = None
        else:
            if (self.held is None) or hold:
                self.current, self.bag = self.generate_minos()
            else:
                self.current = self.held
                self.held = None

    def hold(self):
        if self.held is None:
            self.held = self.current.copy()
            self.pop_from_bag(hold=True)
        else:
            self.step('nop')

    def paste_to_board(self):
        t = self.board.transpose()
        points = self.current.render()
        color = INTS[self.current.type]
        self.pop_from_bag()
        cleared = 0
        for point in points:
            if point[1] < 22:
                self.gameover = True
            self.board[tuple(point)] = color
        for i in range(40):
            if 0 not in t[i]:
                cleared += 1
                self.board[:, :i + 1] = np.concatenate(
                    # Explanation of this black magic:
                    # 1) Take a slice of the array from [(0,0), (10,i)) so
                    #    from y=0 and stop just before the completed row
                    #    self.board[:, :i]
                    # 2) Transpose it, that way it's an array of rows
                    #    self.board[:, :i].transpose()
                    # 3) Create an empty row i.e., an array with one row full of zeros
                    #    np.zeros((1, 10))
                    # 4) Stick the empty row atop the slice of rows
                    #    np.concatenate(
                    #       (np.zeros((1, 10)), self.board[:, :i].transpose())
                    #    )
                    # 5) Transpose the resulting array so that it's an array of columns
                    # 6) Set the board from y=0 to y=i (completed row) to the newly created slice, clearing the
                    #    completed row and shifting the rows
                    (np.zeros((1, 10)), self.board[:, :i].transpose())
                ).transpose()
        cleared = 4 if cleared > 4 else cleared
        self.score += [0, 40, 100, 300, 1200][cleared]

    def render(self):
        buffer = deepcopy(self.board)
        # try:
        for point in self.current.render():
            buffer[tuple(point)] = 8
        # except AttributeError:
            # pass
        return buffer[:, 20:]

    def timetick(self):
        if not self.gameover:
            self.step('down')
            threading.Timer(self.tickdelay, self.timetick).start()

    def observations(self):
        board = self.render()
        board[np.logical_and(board > 0, board < 8)] = 1
        board[board == 8] = 2
        board /= 2

        held = INTS[self.held if self.held is None else self.held.type] / 8
        current = INTS[self.current.type] / 8

        return np.array([current, held, *board.flatten()])

    def neatstep(self):
        self._stepfromactivation(
            self.network.activate(
                self.observations()
            )
        )

    def neatplay(self):
        # self.gametimer.start()
        self.tickdelay = 0.1
        self.timetick()
        while not self.gameover:
            self.neatstep()
        return self.fitness

    def _stepfromactivation(self, activation):
        activation = [x if x > 0 else 0 for x in activation]
        if sum(activation) == 0:
            activation = [1] * 6
        # print(activation)
        choice = choices(
            population = ['left', 'right', 'rotate', 'drop', 'nop', 'hold'],
            weights=activation,
            k=1
        )[0]
        # print(f'{np.round(activation, 3)} -> {choice}')
        self.step(choice)

    @property
    def fitness(self):
        return ln(self.score + 1)

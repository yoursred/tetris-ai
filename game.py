# import pygame
# import pickle
# import time
# import time
# from pprint import pprint
import time

import numpy as np
from numpy import pi, cos, sin, log as ln
from random import choices
from neat.nn.feed_forward import FeedForwardNetwork as FFN
from neat.nn.recurrent import RecurrentNetwork as RN
from random import shuffle
import threading
from typing import Tuple, List
from copy import deepcopy
import pygame

# import inspect
# from uuid import uuid4


MINOS = {  # str to tetromino mapping, with tetrominos being defined
    # as a tuple of points, each point being a square of the mino
    # with (0, 0) being the "center" of it
    'I': ((-2, -1), (-1, -1), (0, -1), (1, -1)),
    'J': ((-1, 0), (-1, -1), (0, 0), (1, 0)),
    'L': ((-1, 0), (1, -1), (0, 0), (1, 0)),
    'O': ((-1, -1), (-1, 0), (0, -1), (0, 0)),
    'S': ((-1, 0), (0, 0), (0, -1), (1, -1)),
    'T': ((-1, 0), (0, 0), (0, -1), (1, 0)),
    'Z': ((-1, -1), (0, -1), (0, 0), (1, 0))
}

ROTATION_ORIGINS = {  # What to rotate the tetrominos around
    'I': (-.5, -.5),
    'J': (0, 0),
    'L': (0, 0),
    'O': (-.5, -.5),
    'S': (0, 0),
    'T': (0, 0),
    'Z': (0, 0)
}

ANGLES = {  # Direction str to angle mapping, in radians
    'up': 0,
    'right': pi / 2,
    'down': pi,
    'left': 3 * pi / 2
}

DIRECTIONS = {  # Direction str to direction str mapping.
    # Used when rotating the piece clockwise
    'up': 'right',
    'right': 'down',
    'down': 'left',
    'left': 'up'
}

WALL_KICKS = {
    #     TODO: Implement wall kicks
}

SPAWN_POSITIONS = {  # Tetromino type str to point mapping.
    # Initial spawn position of said tetromino
    'I': (5, 20),
    'J': (4, 21),
    'L': (4, 21),
    'O': (5, 21),
    'S': (4, 21),
    'T': (4, 21),
    'Z': (4, 21)
}

INTS = {  # Tetromino type str to int mapping.
    # None here being non-existant, only used for current held piece.
    None: 0,
    'I': 1,
    'J': 2,
    'L': 3,
    'O': 4,
    'S': 5,
    'T': 6,
    'Z': 7
}
TOINTS = {  # Reverse mapping of last one.
    # Used when drawing with pygame.
    v: k for k, v in INTS.items()
}

# 2D rotation matrix
R = lambda theta: np.array([[cos(theta), -1 * sin(theta)], [sin(theta), cos(theta)]])


# Rotate point around another
def rotate(point, theta, origin=(0, 0)):
    offset = point - origin
    rotated = R(theta).dot(offset)
    return rotated + origin


# TODO: document


class Mino:
    """
    Tetromino piece class.

    :param type_: Type of Tetromino, can be any of: IJLOSTZ
    :type type_: str
    :param pos: Location of Tetromino, will use a default value if None, defaults to None
    :type pos: Tuple[int, int], optional
    :raises ValueError: When specified type not any of: IJLOSTZ
    """

    def __init__(self, type_: str, pos: Tuple[int, int] = None) -> None:
        if type_ not in 'IJLOSTZ':
            raise ValueError(f'invalid tetromino type {repr(type_)}')
        self.type = type_
        if pos is None:
            pos = SPAWN_POSITIONS[type_]
        self.x: int = pos[0]
        self.y: int = pos[1]
        self.direction = 'up'
        self.pos_log = [('', tuple(self.pos()), self.direction)]

    def render(self):
        """
        Applies relevent rotation, translation to get where every single square is.
        :return: List of points reperesenting squares of Tetromino
        :rtype: List[Tuple[int, int], ...]
        """
        points = map(np.array, MINOS[self.type])  # Turn tuple points to numpy arrays for conveniance
        points = map(  # Rotate around relevant rotation origin
            rotate,
            points,
            (ANGLES[self.direction],) * 4,
            (ROTATION_ORIGINS[self.type],) * 4
        )
        points = map(  # Round, cast to int then translate by current position
            lambda x: np.round(x).astype(int) + self.pos(),  # numpy doesn't round when casting to int
            # https://stackoverflow.com/a/43920513/16338589
            points
        )
        return list(points)

    def rotate(self) -> None:
        """
        Rotate clockwise.
        :return: None
        """
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
            if (point[0] + dirx not in range(10)) or (point[1] + diry not in range(40)):
                return True
            elif board[point[0] + dirx, point[1] + diry] != 0:
                return True
        return False

    def move(self, board, dir_):
        # self._log_id()
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
        elif dir_ == 'down':  # or dir_ == 'nop':
            if not self.check_collision(board, 0, 1):
                self.y += 1
                return False
            return True
        elif dir_ == 'drop':
            while not self.check_collision(board, 0, 1):
                self.y += 1
            return True
        elif dir_ == 'nop':
            pass


class Game(object):
    def __init__(self, board: np.array = None, network: (FFN, RN) = None, render=False, hyper=False):

        # self.uuid = str(uuid4())

        self.unbag = []

        if board is None:
            self.board = np.zeros((10, 40))
        else:
            self.board = board
        self.network = network
        current, bag = self.generate_minos()
        self.current: Mino = current
        self.bag0: List[Mino] = bag
        minos = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        shuffle(minos)
        self.bag1: List[Mino] = [*map(Mino, minos)]
        self.held = None
        self.score = 0
        self.penalties = 0
        self.gameover = False
        self.actions = []
        self.placed_pieces = 0
        self.last_cleared_lines = 0

        self.ticking = False
        self.stepping = False
        self.tickdelay = 1
        # self.timer = threading.Timer(self.tickdelay, self.timetick)
        self.timer = None

        self.started_at = 0

        self.hyper = hyper

        self.render = render
        if render:
            pygame.init()
            self.display = pygame.display.set_mode((1000, 800))
            pygame.display.set_caption('Tetris')
            self.font = pygame.font.SysFont('Ubuntu Mono', size=40)
        self._moving = {
            'I': (0, 255, 255),
            'J': (0, 0, 255),
            'L': (255, 165, 0),
            'O': (255, 255, 0),
            'S': (0, 255, 0),
            'T': (128, 0, 128),
            'Z': (255, 0, 0)
        }
        self.possible_actions = ['left:  ', 'right: ', 'rotate:', 'drop:  ', 'nop:   ', 'hold:  ']

    def paint(self, choice, activation=None, weights=None, timearg=0):
        # print(activation)
        if activation is None:
            activation = []
        if weights is None:
            weights = []
        buffer = self.render_board()
        self.display.fill((200, 200, 200))
        test = self.font.render(f'chosen: {choice}', True, (80, 39, 39))
        for i, val in enumerate(activation):
            # print(activation)
            text = self.font.render(f'{self.possible_actions[i]} {round(val, 7)}', True, (80, 39, 39))
            self.display.blit(text, (16, i * 40))
        self.display.blit(test, (16, 14 * 40))

        # for i, val in enumerate(weights):
        #     # print(activation)
        #     text = self.font.render(f'w{i} {round(val, 7)}', True, (80, 39, 39))
        #     self.display.blit(text, (16, i*40))

        score = self.font.render(f'score:  {self.score}', True, (80, 39, 39))
        self.display.blit(score, (16, 11 * 40))

        penalties = self.font.render(f'pnlt:   {self.penalties}', True, (80, 39, 39))
        self.display.blit(penalties, (16, 12 * 40))

        fitness = self.font.render(f'ftnss: {" " if self.fitness >= 0 else ""}{round(self.fitness, 5)}', True,
                                   (80, 39, 39))
        self.display.blit(fitness, (16, 13 * 40))

        _time = self.font.render(f'time:   {round(timearg, 5)}', True, (80, 39, 39))
        self.display.blit(_time, (16, 15 * 40))

        held = self.font.render('   HELD', True, (80, 39, 39))
        self.display.blit(held, (800, 0))
        bag = self.font.render('    BAG', True, (80, 39, 39))
        self.display.blit(bag, (800, 160))
        if self.held is not None:
            piece = self.held.copy()
            piece.x = 1
            piece.y = 3

            for x, y in piece.render():
                sx = ((x + 2) * 30) + 800
                sy = (y * 30)
                pygame.draw.rect(self.display, self._moving[piece.type], pygame.Rect(sx, sy, 30, 30))
                pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 30, 30), 2)

        for i, piece in enumerate(self.bag0[::-1]):
            piece = piece.copy()
            piece.x = 1
            piece.y = i * 3 + 10
            for x, y in piece.render():
                sx = ((x + 2) * 30) + 800
                sy = (y * 30)
                pygame.draw.rect(self.display, self._moving[piece.type], pygame.Rect(sx, sy, 30, 30))
                pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 30, 30), 2)

        for x in range(10):
            for y in range(20):
                sx = (x * 40) + 400
                sy = (y * 40)
                pygame.draw.rect(self.display, (0, 28, 70), pygame.Rect(sx, sy, 40, 40))
                pygame.draw.rect(self.display, (0, 35, 80), pygame.Rect(sx, sy, 40, 40), 2)
        for x in range(10):
            for y in range(20):
                sx = (x * 40) + 400
                sy = (y * 40)
                if buffer[x, y] == 0:
                    pass
                elif buffer[x, y] in range(1, 8):
                    pygame.draw.rect(self.display, self._moving[TOINTS[buffer[x, y]]], pygame.Rect(sx, sy, 40, 40))
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 40, 40), 2)
                elif buffer[x, y] == 8:
                    pygame.draw.rect(self.display, self._moving[self.current.type], pygame.Rect(sx, sy, 40, 40))
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 40, 40), 2)
        pygame.display.update()

    def game_step(self, cmd='nop'):
        if cmd not in ('nop', 'left', 'right', 'rotate', 'hold', 'drop', 'down', 'gravity'):
            raise ValueError(f'invalid command: {repr(cmd)}')
        self._penalize(cmd)
        if cmd == 'gravity':
            if self.current.move(self.board, 'down') and not self.stepping:
                self.paste_to_board()

        else:
            self.stepping = True
            if cmd == 'nop':
                pass
            elif cmd != 'hold':
                if self.current.move(self.board, cmd):
                    self.paste_to_board()
                    self.actions.append(cmd)
            elif cmd == 'hold':
                self.hold()
                self.actions.append(cmd)
            self.stepping = False
        self.timetick()

    @staticmethod
    def generate_minos():
        minos = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        # minos = ['O']
        shuffle(minos)
        for i, mino in enumerate(minos):
            minos[i] = Mino(mino)
        current, *minos = minos
        return current, minos

    def pop_from_bag(self, hold=False):
        if self.bag1:
            if (self.held is None) or hold:
                self.current = self.bag0.pop()
                self.bag0.insert(0, self.bag1.pop())
            else:
                self.current = self.held
                self.held = None
        else:
            if (self.held is None) or hold:
                minos = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
                shuffle(minos)
                self.bag1 = [*map(Mino, minos)]
                self.current = self.bag0.pop()
                self.bag0.append(self.bag1.pop())
            else:
                self.current = self.held
                self.held = None

    def hold(self):
        # pass
        if self.held is None:
            self.held = self.current.copy()
            # self.held.pos_log = self.current.pos_log.copy()
            self.pop_from_bag(hold=True)
        else:
            self.game_step('nop')
            # pass

    def paste_to_board(self):
        t = self.board.transpose()
        points = self.current.render()
        color = INTS[self.current.type]
        self.unbag.append(self.current)
        self.pop_from_bag()
        cleared = 0
        for point in points:
            if point[1] < 22:
                self.gameover = True
            self.board[tuple(point)] = color
        for i in range(40):
            if 0 not in t[i]:
                cleared += 1
                # print('cleared!')
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
        self.last_cleared_lines = cleared
        cleared = 4 if cleared > 4 else cleared
        self.score += [0, 40, 100, 300, 1200][cleared]
        self.placed_pieces += 1

    def render_board(self):
        buffer = deepcopy(self.board)
        for point in self.current.render():
            buffer[tuple(point)] = 8
        return buffer[:, 20:]

    def timetick(self):
        self.ticking = True
        if self.timer is not None:
            if self.timer.is_alive():
                return
        if not self.gameover:
            # print('Say wa')
            self.timer = threading.Timer(self.tickdelay, self.timetick)
            self.timer.start()
            self.game_step('gravity')

            # self.timer.join()

    def observations(self):
        board = self.render_board()
        board[np.logical_and(board > 0, board < 8)] = 1
        board[board == 8] = 2
        board /= 2

        held = INTS[self.held if self.held is None else self.held.type] / 7
        current = INTS[self.current.type] / 7

        header = np.array([held,
                           current,
                           *[INTS[x.type] / 8 for x in self.bag0[::-1]], *(0,) * (8 - len(self.bag0))]
                          ).reshape((1, 10))

        return np.concatenate(
            (header,
             board.transpose())
        ).transpose()

    def neatstep(self):
        activation = self.network.activate(
            self.observations().flatten()
        )

        choice = self._stepfromactivation(activation)
        self.actions.append(choice)
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

    def neatppstep(self, network, depth):
        network.Flush()
        network.Input(self.observations().flatten())  # can input numpy arrays, too
        # for some reason only np.float64 is supported
        for _ in range(depth):
            network.Activate()
        activation = network.Output()
        print(list(activation))

        choice = self._stepfromactivation(activation)
        self.actions.append(choice)
        self.current.pos_log.append((choice, tuple(self.current.pos()), self.current.direction))
        if self.render:
            self.paint(choice, activation)

    def neatppplay(self, network, depth, tickdelay=0.1, timeout=60):
        self.started_at = time.time()
        self.tickdelay = tickdelay
        self.timetick()
        while (not self.gameover) and time.time() - self.started_at < timeout:
            self.neatppstep(network, depth)
        return self.fitness

    def _stepfromactivation(self, activation):
        if not self.ticking:
            self.timetick()
        # print(activation)
        # activation = [x if x > 0 else 0 for x in activation]
        if sum(activation) == 0:
            activation = [1] * 6
        choice = choices(
            population=['left', 'right', 'rotate', 'drop', 'nop', 'hold'],
            weights=activation,
            k=1
        )[0]
        # print(f'{np.round(activation, 3)} -> {choice}')
        self.game_step(choice)
        return choice

    def generate_legal_moves(self):
        moves = []

        for n, rot in enumerate(DIRECTIONS):
            test = self.current.copy()
            test.x, test.y = self.current.pos()
            test.direction = rot
            for i in range(10 - max(test.render(), key=lambda x: x[0])[0]):
                moves.append((
                    *('rotate',) * n,
                    *('right',) * i,
                    'drop'))
            for i in range(min(test.render(), key=lambda x: x[0])[0]):
                moves.append((
                    *('rotate',) * n,
                    *('left',) * (i + 1),
                    'drop'))
        return moves

    def pick_safest_move(self):
        moves = {move: self.evaluate_move(move) for move in self.generate_legal_moves()}

        return max(moves.items(), key=lambda x: x[1])[0]

    def evaluate_move(self, move):
        copy = deepcopy(self)
        copy.gameover = True
        for _ in move:
            copy.game_step(_)
        return copy.tree_fitness() - self.tree_fitness()

    def tree_play(self, tickdelay=0.01, timeout=6, depth=0):
        self.started_at = time.time()
        self.tickdelay = tickdelay
        self.timetick()
        # print(timeout)
        while (not self.gameover) and time.time() - self.started_at < timeout:
            # print(time.time() - self.started_at)
            for move in self.pick_safest_move():
                self.game_step(move)
                if self.render:
                    self.paint(move, timearg=(time.time() - self.started_at))
        return self.fitness

    def _penalize(self, action):
        if action == 'rotate' and self.current.type == 'O':
            self.penalties += 5
        if action == 'left' and self.current.check_collision(self.board, -1, 0):
            self.penalties += 5
        if action == 'right' and self.current.check_collision(self.board, 1, 0):
            self.penalties += 5
        if action == 'nop' and self.actions[-60:].count('nop') >= 25:
            self.penalties += 5
        # if action == 'rotate' and self.actions[-5:].count('rotate') > 3:
        #     self.penalties += 5
        if action == 'hold' and self.held is not None:
            self.penalties += 5
        if self.bag0:
            if action == 'hold' and self.current.type == self.bag0[-1].type:
                self.penalties += 5
        if action in ('right', 'left') \
                and self.actions[-60:].count('left') >= 25 \
                and self.actions[-60:].count('right') >= 25:
            self.penalties += 10
        return action

    @property
    def shape_fitness(self):
        best = max(map(np.sum, self.observations().transpose()[1:]))
        return best

    @property
    def fitness(self):
        # if self.score == 0:
        # self.score = (self.endtime - self.starttime) ** (1/2)
        # score = np.sqrt(self.score) * 25
        # return score - np.sqrt(self.penalties)

        # return np.sqrt(self.placed_pieces) + score
        return self.score - np.sqrt(self.penalties)

    def tree_fitness(self):

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

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ('timer', 'display', 'font'):
                setattr(result, k, deepcopy(v, memodict))
            result.timer = None
            result.render = False
        return result

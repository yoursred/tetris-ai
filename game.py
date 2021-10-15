from random import shuffle
import threading
from typing import Tuple, List
from copy import deepcopy

from neat.nn.feed_forward import FeedForwardNetwork as FFN
from neat.nn.recurrent import RecurrentNetwork as RN
import pygame
import numpy as np

from constants import *
from utils import rotate


class Mino:
    """
    Tetromino piece class.

    :param type_: Type of-   Tetromino, can be any of: IJLOSTZ
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
        """
        Check collision of this piece in `board` relative to `(dirx, diry)`
        :param board: Tetris board in the form of a numpy array
        :type dirx: int
        :type diry: int
        :rtype: bool
        """
        # Check every block in the Tetromino
        for point in self.render():
            # Is it within bounds?
            if (point[0] + dirx not in range(10)) or (point[1] + diry not in range(40)):
                return True
            # Does it occupy an empty space?
            elif board[point[0] + dirx, point[1] + diry] != 0:
                return True
        return False

    def move(self, board, dir_):
        """
        Move or rotate the Tetromino with respect to the provided `board` argument
        :param board: Tetris board in the form of a numpy array
        :param dir_: Could be any of left, right, rotate, down, drop, nop
        :type dir_: str
        :return: None
        """
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
    """
    This object represents a Tetris game and implements most of its basic functionality (looking at you wallkicks)

    :param board: Numpy array representing the playfield, only specify for replaying or changing the size of the board
    :type board: np.array
    :param network: Neural network with an `activate` method
    :type network: Any
    :param render: Boolean specifying whether to render the game with `pygame` or not
    :type render: bool
    """
    def __init__(self, board: np.array = None, network: (FFN, RN) = None, render=False):
        if board is None:
            self.board = np.zeros((10, 40))
        else:
            self.board = board
        self.network = network
        current, *bag = self.generate_minos()
        self.current: Mino = current
        self.bag0: List[Mino] = bag
        self.bag1: List[Mino] = self.generate_minos()
        self.held = None
        
        self.score = 0
        self.gameover = False
        
        self.placed_pieces = 0
        self.last_cleared_lines = 0

        self.ticking = False
        self.stepping = False
        self.tickdelay = 1
        self.timer = None
        self.started_at = 0
        self.time_left = 0

        self.render_timer = None
        self.framerate = 30
        self.render = render
        if render:
            pygame.init()
            self.display = pygame.display.set_mode((1000, 800))
            pygame.display.set_caption('Tetris')
            self.font = pygame.font.SysFont('Ubuntu Mono', size=40)

        self.possible_actions = ['left:  ', 'right: ', 'rotate:', 'drop:  ', 'nop:   ', 'hold:  ']

    @staticmethod
    def generate_minos():
        """
        Generate a shuffled list of `Mino` objects
        :return:
        :rtype: list[Mino]
        """
        minos = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        # minos = ['O']
        shuffle(minos)
        for i, mino in enumerate(minos):
            minos[i] = Mino(mino)
        # current, *minos = minos
        return minos

    def game_step(self, cmd='nop'):
        """
        Execute an action in the game, this can be movement (left, right, down), rotation (clockwise), holding, dropping
        , nop (no operation), or "gravity".
        Gravity is only used by the `Game.timetick` method, it checks if an action is in the middle of execution and
        waits for it to finish, as `Game.timetick` is asynchronous.
        :param cmd: The action to execute, any of ('nop', 'left', 'right', 'rotate', 'hold', 'drop', 'down', 'gravity')
        :return: None
        """
        if cmd not in ('nop', 'left', 'right', 'rotate', 'hold', 'drop', 'down', 'gravity'):
            raise ValueError(f'invalid command: {repr(cmd)}')

        if self.render and not self.stepping:
            self.paint(cmd)

        if cmd == 'gravity':
            if self.current.move(self.board, 'down') and not self.stepping:
                self.paste_to_board()
            self.timetick()
        else:
            self.stepping = True  # Acquire a lock on execution to prevent gravity from taking effect
            if cmd == 'nop':
                pass
            elif cmd != 'hold':
                if self.current.move(self.board, cmd):
                    self.paste_to_board()
            elif cmd == 'hold':
                self.hold()
            self.stepping = False

        if self.render:
            self.paint(cmd)

    def pop_from_bag(self, hold=False):
        """
        Remove the last tetromino from the bag onto the playfield
        :param hold: Ignore the held piece
        :type hold: bool
        :return: None
        """
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
        """
        Hold the current piece
        :return: None
        :rtype: None
        """
        if self.held is None:
            self.held = self.current.copy()
            self.pop_from_bag(hold=True)

    def paste_to_board(self):
        """
        Make the current piece a part of the playfield
        :return: None
        :rtype: None
        """
        t = self.board.transpose()
        points = self.current.render()
        color = INTS[self.current.type]
        self.pop_from_bag()
        cleared = 0
        for point in points:
            if point[1] < 22:
                # If any block of the piece is above y=22, the game is lost
                self.gameover = True
            self.board[tuple(point)] = color
        for i in range(40):
            if 0 not in t[i]:
                cleared += 1
                self.board[:, :i + 1] = np.concatenate(
                    # Explanation of this black magic fuckery:
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
        self.time_left += 0.75 * cleared
        cleared = 4 if cleared > 4 else cleared
        self.score += [0, 40, 100, 300, 1200][cleared]
        self.placed_pieces += 1

    def timetick(self):
        """
        Initiate the asynchronous gravity clock.
        :return: None
        :rtype: None
        """
        self.ticking = True
        if self.timer is not None:
            if self.timer.is_alive():
                return
        if not self.gameover:
            self.timer = threading.Timer(self.tickdelay, self.timetick)
            self.timer.start()
            self.game_step('gravity')

    def display_tick(self):
        """
        Initiate the asynchronous rendering clock.
        :return: None
        :rtype: None
        """
        self.render_timer = threading.Timer(1 / self.framerate, self.display_tick)
        self.render_timer.start()
        self.paint('')

    def render_board(self):
        """
        Generate a drawable playfield that includes the current piece.
        :return: `np.array` with blocks of the current pieces marked with 8
        :rtype: np.array
        """
        buffer = deepcopy(self.board)
        for point in self.current.render():
            buffer[tuple(point)] = 8
        return buffer[:, 20:]

    def paint(self, choice, timearg=0.0, extra_info=None):
        """
        Big fat function that draws the game UI using pygame.
        "This is so fat that it didn't was was a McDonald's Big Mac" -Some person, probably.
        :param choice: The action chosen by whatever is playing the game. This argument will removed as it does not care
        doesn't allow reading the choice very easily with the current system. TODO remove and refractor
        :type choice: str
        :param timearg: How much time is left or spent. Again, it would be better to remove this to declutter the
        function call. TODO remove and refractor
        :type timearg: float, int
        :param extra_info: A dictionary containing relevant information to the current player. TODO implement
        :type extra_info: dict[str, str]
        :return: None
        :rtype: None
        """
        if extra_info is None:
            # noinspection PyUnusedLocal
            extra_info = {}

        buffer = self.render_board()
        self.display.fill((200, 200, 200))
        test = self.font.render(f'chosen: {choice}', True, (80, 39, 39))
        self.display.blit(test, (16, 14 * 40))

        score = self.font.render(f'score:  {self.score}', True, (80, 39, 39))
        self.display.blit(score, (16, 11 * 40))

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
                pygame.draw.rect(self.display, MINO_COLORS[piece.type], pygame.Rect(sx, sy, 30, 30))
                pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 30, 30), 2)

        for i, piece in enumerate(self.bag0[::-1]):
            piece = piece.copy()
            piece.x = 1
            piece.y = i * 3 + 10
            for x, y in piece.render():
                sx = ((x + 2) * 30) + 800
                sy = (y * 30)
                pygame.draw.rect(self.display, MINO_COLORS[piece.type], pygame.Rect(sx, sy, 30, 30))
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
                    pygame.draw.rect(self.display, MINO_COLORS[TOINTS[buffer[x, y]]], pygame.Rect(sx, sy, 40, 40))
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 40, 40), 2)
                elif buffer[x, y] == 8:
                    pygame.draw.rect(self.display, MINO_COLORS[self.current.type], pygame.Rect(sx, sy, 40, 40))
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(sx, sy, 40, 40), 2)
        pygame.display.update()

    def observations(self):
        """
        Gives a more computer-vision friendly version of the playfield.
        The top row shows the types of the current piece and bag respectively.
        `0/2` means empty, `1/2` occupied by a fixed block, and `2/2` is occupied by the current piece.
        :return:
        :rtype: np.array
        """
        board = self.render_board()
        board[np.logical_and(board > 0, board < 8)] = 1  # Reduce all colors to 1
        board[board == 8] = 2  # Blocks of the current piece
        board /= 2  # Normalize

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

    @property
    def fitness(self):
        """
        :return: Agent fitness
        :rtype: float
        """
        return float(self.score)

    # noinspection PyArgumentList
    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ('timer', 'display', 'font', 'render_timer'):
                setattr(result, k, deepcopy(v, memo=memodict))
            result.timer = None
            result.render = False
        return result

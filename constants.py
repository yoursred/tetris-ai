import numpy as np

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

MINO_COLORS = {
    'I': (0, 255, 255),
    'J': (0, 0, 255),
    'L': (255, 165, 0),
    'O': (255, 255, 0),
    'S': (0, 255, 0),
    'T': (128, 0, 128),
    'Z': (255, 0, 0)
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
    'right': np.pi / 2,
    'down': np.pi,
    'left': 3 * np.pi / 2
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
    # None here being non-existent, only used for current held piece.
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

del np

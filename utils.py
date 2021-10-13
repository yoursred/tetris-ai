import numpy as np

"""
This file contains miscellaneous functions.
"""


def r_matrix_2d(theta):
    return np.array([[np.cos(theta), -1 * np.sin(theta)],
                     [np.sin(theta),      np.cos(theta)]])


# Rotate point around another
def rotate(point, theta, origin=(0, 0)):
    offset = point - origin
    rotated = r_matrix_2d(theta).dot(offset)
    return rotated + origin

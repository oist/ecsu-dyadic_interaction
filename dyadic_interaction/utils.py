"""
TODO: Missing module docstring
"""

import string
import numpy as np
from numpy import pi

# CONSTANTS
TWO_PI = 2*pi
RANDOM_CHAR_SET = string.ascii_uppercase + string.digits

def random_string(size=5):
    '''
    generates random alphanumeric string
    '''
    return ''.join(np.random.choice(RANDOM_CHAR_SET) for _ in range(size))


def modulo_radians(theta):
    '''
    ensure that the angle is between 0 and 2*pi
    0 <= result_theta < 2*pi
    '''
    return theta % TWO_PI


def rotate_cw_matrix(theta):
    '''
    Returns a rotation clock-wise matrix
    e.g.,
    v = np([np.sqrt(3),1])  # vector of length 2 at 30 (pi/6) angle
    theta = pi/6
    r = rotate_cw_matrix(theta)
    # --> np.dot(v,r) = [2,0]
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

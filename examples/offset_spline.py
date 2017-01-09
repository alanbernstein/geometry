#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from spline import slope_controlled_bezier_curve
from curves import offset_curve, add_frenet_offset_2D

from panda.debug import debug


def offset_spline_example():
    width = 1.0

    knots = np.array([
        [+1.5, +1.5, -0.0, -1.5],
        [+0.0, -0.0, -0.5, -0.0],
        [-1.5, +1.5, -0.5, +1.0],
        # [-1.0, +5.0, +0.0, +1.0],
        [-3.0, +9.0, +0.0, +3.0],
        [+0.0, +12.0, +0.5, +0.0],
        [+3.0, +9.0, +0.0, -3.0],
    ])

    xy = slope_controlled_bezier_curve(knots)

    t = np.zeros((width, len(xy)))
    n = np.ones((width, len(xy)))
    tn = np.vstack((t, n)).T
    # debug()
    xyo0 = add_frenet_offset_2D(xy, tn)
    xyo1 = add_frenet_offset_2D(xy, -tn)

    plt.plot(xy[:, 0], xy[:, 1], 'k-', label='bezier curve')
    plt.plot(xyo0[:, 0], xyo0[:, 1], 'b-', label='offset curve')
    plt.plot(xyo1[:, 0], xyo1[:, 1], 'c-', label='offset curve')

    for x, y, mx, my in knots:
        plt.plot(x, y, 'ro')
        plt.plot([x, x + mx], [y, y + my], 'r-')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

offset_spline_example()

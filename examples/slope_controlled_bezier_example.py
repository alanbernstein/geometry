#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from spline import slope_controlled_bezier_curve


def slope_controlled_bezier_example():
    s = .25
    d = 1
    knots = np.array([[0 * d, +1.0, s, 0],
                      [1 * d, -0.9, s, 0],
                      [2 * d, +0.8, s, 0],
                      [3 * d, -0.7, 0, 5 * s]])

    xy = slope_controlled_bezier_curve(knots)

    plt.plot(xy[:, 0], xy[:, 1], 'k-', label='bezier curve')
    for x, y, mx, my in knots:
        plt.plot(x, y, 'ro')
        plt.plot([x, x + mx], [y, y + my], 'r-')
    plt.show()


slope_controlled_bezier_example()

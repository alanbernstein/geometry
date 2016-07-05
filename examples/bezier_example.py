#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from spline import bezier_curve


def bezier_example():
    knots = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    xy = bezier_curve(knots)

    plt.plot(xy[:, 0], xy[:, 1])
    plt.plot(knots[:, 0], knots[:, 1], "ro-")
    for nr in range(len(knots)):
        plt.text(knots[nr][0], knots[nr][1], nr)
    plt.axis([-0.05, 1.05, -0.05, 2.05])
    plt.show()


bezier_example()

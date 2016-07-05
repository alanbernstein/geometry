#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

from curves import naturalize_parameter


def archimedean_spiral(t):
    r = 2 * t
    th = t * 2 * np.pi
    x = r * np.cos(th)
    y = r * np.sin(th)
    return np.vstack((x, y)).T


def frenet_example_2D():
    t = np.linspace(0, 2, 1201)
    base = archimedean_spiral(t)
    base_natural, _, _ = naturalize_parameter(base)

    # plot curve and equally-spaced points using natural parameterization
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    plt.plot(base[:, 0], base[:, 1], 'k-')
    num_pts = 12
    step = int((len(t) - 1) / num_pts)
    plt.plot(base_natural[::step, 0], base_natural[::step, 1], 'ko')
    plt.show()


if __name__ == '__main__':
    frenet_example_2D()

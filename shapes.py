#!/usr/bin/python
import numpy as np

from panda.plot_utils import qplot
from panda.debug import debug
import matplotlib.pyplot as plt


def main():
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    for d in np.linspace(180, 225, 6):
        print(d)
        c = heart_curve_square(arc_degrees=d)
        plt.plot(c[:, 0], c[:, 1])
    plt.show(block=False)

    c = heart_curve_square_180()
    plt.plot(c[:, 0], c[:, 1], 'k--')
    c = heart_curve_square_225()
    plt.plot(c[:, 0], c[:, 1], 'k--')

    debug()


def square(c=0, s=1):
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


# cc = np.exp(np.linspace(0, 2j*np.pi, 65))
# circle = np.vstack((cc.real, cc.imag)).T
def arc(c=0, r=1, a1=0, a2=2*np.pi):
    # center, radius angle1, angle2
    # defaults -> unit circle

    # TODO: this is super clunky, there must be a better way?
    if type(c) == np.ndarray:
        if c.shape == (2,):
            c = c[0] + c[1]*1j
        elif c.shape == (1, 2):
            c = c[:, 0] + c[:, 1]*1j
        else:
            debug()

    cc = c + r * np.exp(1j * np.linspace(a1, a2, 65))
    return np.vstack((cc.real, cc.imag)).T


def circle(c=0, r=1):
    return arc(c, r)


def heart_curve_pointy(num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points + 1)
    x = np.sin(t) ** 3
    y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16
    return np.vstack((x, y)).T


def heart_curve_square(num_points=100, arc_degrees=180):
    """arc_degrees = angular length of one curved part of heart, in [180, 225]"""
    t = np.linspace(0, float(arc_degrees) / 180 * np.pi, num_points + 1)

    x_arc = np.cos(t - np.pi / 4)
    y_arc = np.sin(t - np.pi / 4)
    x_offset = -x_arc[-1]
    y_offset = 0
    y_bottom = -x_arc[0] + y_arc[0] + x_arc[-1]

    x0 = x_arc + x_offset
    y0 = y_arc + y_offset
    x1 = -x_arc[::-1] - x_offset
    y1 = y_arc[::-1] + y_offset
    x2 = [0, x0[0]]
    y2 = [y_bottom, y0[0]]

    x = np.hstack((x0, x1, x2))
    y = np.hstack((y0, y1, y2))

    return np.vstack((x, y)).T


# almost deprecated, but the new one doesnt quite match yet
r2 = np.sqrt(2)
def heart_curve_square_225(num_points=100):
    t = np.linspace(0, 1.25 * np.pi, num_points + 1)
    x0 = np.cos(t - np.pi / 4) + 1
    y0 = np.sin(t - np.pi / 4)
    x1 = np.cos(t) - 1
    y1 = np.sin(t)
    x2 = [0, 1 + r2 / 2]
    y2 = [-1 - r2, -r2 / 2]

    x = np.hstack((x0, x1, x2))
    y = np.hstack((y0, y1, y2))

    return np.vstack((x, y)).T


def heart_curve_square_180(num_points=100):
    t = np.linspace(0, np.pi, num_points + 1)

    x0 = np.cos(t - np.pi / 4) + np.sqrt(2) / 2
    y0 = np.sin(t - np.pi / 4)
    x1 = np.cos(t + np.pi / 4) - np.sqrt(2) / 2
    y1 = np.sin(t + np.pi / 4)
    x2 = [0, r2]
    y2 = [-r2 - np.sqrt(2) / 2, -np.sqrt(2) / 2]

    x = np.hstack((x0, x1, x2))
    y = np.hstack((y0, y1, y2))

    return np.vstack((x, y)).T

if __name__ == '__main__':
    main()

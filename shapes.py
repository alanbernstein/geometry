#!/usr/bin/python
import numpy as np

from panda.plot_utils import qplot
from panda.debug import debug
import matplotlib.pyplot as plt


def main():
    c = heart_curve_square_180()
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    qplot(c, 'r-')
    debug()


def heart_curve_pointy(num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points + 1)
    x = np.sin(t) ** 3
    y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16
    return np.vstack((x, y)).T


def heart_curve_square(num_points=100, arc_degrees=180):
    t = np.linspace(0, arc_degrees / 180 * np.pi, num_points + 1)
    x0 = np.cos(t - np.pi / 4) + 1
    y0 = np.sin(t - np.pi / 4)


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
    y0 = np.sin(t - np.pi / 4) + np.sqrt(2) / 2
    x1 = np.cos(t + np.pi / 4) - np.sqrt(2) / 2
    y1 = np.sin(t + np.pi / 4) + np.sqrt(2) / 2
    x2 = [0, r2]
    y2 = [-r2, 0]

    x = np.hstack((x0, x1, x2))
    y = np.hstack((y0, y1, y2))

    return np.vstack((x, y)).T

if __name__ == '__main__':
    main()

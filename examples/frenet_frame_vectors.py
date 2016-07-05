#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

from curves import frenet_frame


def archimedean_spiral(t):
    r = 2 * t
    th = t * 2 * np.pi
    x = r * np.cos(th)
    y = r * np.sin(th)
    return np.vstack((x, y)).T


def frenet_example_2D():
    t = np.linspace(0, 2, 1201)
    base = archimedean_spiral(t)
    T, N, B = frenet_frame(base)

    # plot curve and frenet frame at subsampled points
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    plt.plot(base[:, 0], base[:, 1], 'k-')
    num_vecs = 12
    step = int((len(t) - 1) / num_vecs)
    for p, t0, n0 in zip(base, T, N)[::step]:
        txy = np.vstack((p, p + t0[0:2]))
        plt.plot(txy[:, 0], txy[:, 1], 'r-')
        nxy = np.vstack((p, p + n0[0:2]))
        plt.plot(nxy[:, 0], nxy[:, 1], 'g-')
    plt.show()


if __name__ == '__main__':
    frenet_example_2D()

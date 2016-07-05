#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

from curves import add_frenet_offset_2D

def diff_curve_natural(t):
    # using a parameter that matches two identical trajectories in both space and spacetime
    x = t
    y = -((1/3.0 <= t) & (t < 2/3.0)).astype(float) / 3.0 * .2
    return np.vstack((x, y)).T
    

def diff_curve_unnatural(t):
    # using a parameter that matches two identical trajectories in space, but not spacetime
    ti = np.linspace(0, 1, 6)
    xi = np.array([0, 1, 1, 2, 2, 3])/3.0
    yi = -np.array([0, 0, 1, 1, 0, 0])/3.0 * .2

    x = np.interp(t, ti, xi)
    y = np.interp(t, ti, yi)
    return np.vstack((x, y)).T


def base_curve_quadratic(t):
    x = t
    y = 0.25 * t * (1-t)
    return np.vstack((x, y)).T
        

def frenet_offset_example_2D():
    # note the shape curve is negative in the null_base plot
    # this is because the normal vector points into the curve
    t = np.linspace(0, 1, 301)

    null_base = np.vstack((t, 0*t)).T
    base = base_curve_quadratic(t)
    shape = diff_curve_natural(t)
    diff = shape - null_base
   
    xy = add_frenet_offset_2D(base, diff)

    fig = plt.figure()
    fig.add_subplot(211, aspect='equal')
    plt.plot(xy[:, 0], xy[:, 1])
    plt.plot(base[:, 0], base[:, 1], 'r--')

    fig.add_subplot(212, aspect='equal')
    plt.plot(shape[:, 0], shape[:, 1])
    plt.plot(null_base[:, 0], null_base[:, 1], 'r--')
    plt.show()


if __name__ == '__main__':
    frenet_offset_example_2D()

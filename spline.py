#!/usr/bin/python
from scipy.misc import comb
import numpy as np

# http://stackoverflow.com/a/12644499/137838


def bernstein_poly(i, n, t):
    """Bernstein polynomial b_{n, i}, as a function of t"""
    return comb(n, i) * ((1 - t) ** (n - i)) * t ** i


# TODO: cache polynomial for fixed nt
# TODO: extend tp N dimensions
# TODO: evaluate via subdivision rather than linear
# TODO: support multiple orders - not clear how that works with slope_controlled
def bezier_curve(knots, nt=1000):
    """Return bezier curve corresponding to given control points
    knots: list of [x, y]
    nt: number of time steps
    return value: (nt+1)x2 array
    See http://processingjs.nihongoresources.com/bezierinfo/"""

    nknots = len(knots)
    xknots = np.array([p[0] for p in knots])
    yknots = np.array([p[1] for p in knots])

    t = np.linspace(0.0, 1.0, nt + 1)

    polynomial_array = np.array([bernstein_poly(i, nknots - 1, t) for i in range(0, nknots)])

    x = np.dot(xknots, polynomial_array)
    y = np.dot(yknots, polynomial_array)

    return np.vstack((x, y)).T


def slope_controlled_bezier_curve(knots, pts_per_segment=50):
    """knots: list of [x, y, mx, my] - my/mx = slope, hypot(mx, my) = power
    pts_per_segment = number of time steps per segment
    return value: K x 2 array"""
    xy = []
    num_segments = len(knots) - 1
    # generate list of segment points
    for n in range(num_segments):
        # assemble slope-knots into basic knots
        p0 = [knots[n][0:2],
              knots[n][0:2] + knots[n][2:4],
              knots[n + 1][0:2] - knots[n + 1][2:4],
              knots[n + 1][0:2]]
        xy0 = bezier_curve(p0, pts_per_segment)

        # remove final point, for all but the last segment
        if n < num_segments - 1:
            xy0 = xy0[0:-1, :]

        xy.append(xy0)

    # combine segments
    return np.vstack(xy)

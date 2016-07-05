#!/usr/bin/python
import numpy as np


def add_frenet_offset_2D(base_curve, offset_vector, base_T=None, base_N=None):
    """apply 2D offset_vector using (T, N) of base_curve as the (x, y) plane"""
    # allow inputting T, N because they can be computed more accurately
    # in the caller if they are segments of a longer curve

    if base_T is None or base_N is None:
        base_T, base_N, base_B = frenet_frame(base_curve)

    xy = []
    for b, d, t, n in zip(base_curve, offset_vector, base_T, base_N):
        new = b + d[0] * t[0:2] + d[1] * n[0:2]
        xy.append(new)

    return np.vstack(xy)


def add_frenet_offset(base_curve, offset_vector, base_T=None, base_N=None, base_B=None):
    """apply 3D offset vector using (T, N, B) of base_curve as the (x, y, z) basis"""
    # TODO: next time you need to use frenet_conform and the above function doesn't work, try this one instead, fix it if broken
    # TODO: do this concisely in numpy
    if base_T is None or base_N is None:
        base_T, base_N, base_B = frenet_frame(base_curve)

    xyz = []
    for base, d, t, n, b in zip(base_curve, offset_vector, base_T, base_N, base_B):
        new = base + d[0] * t + d[1] * n + d[2] * b
        xyz.append(new)

    return np.vstack(xyz)


def naturalize_parameter(r, t=None):
    """given position vector r as a function of t
    return natural parameterization of curve ri
    that is, points of ri are equally spaced along the curve
    ri, si, ti = naturalize(r, t)
    ri = new position curve
    si = new arclength as function of t (should be linear)
    ti = new t parameter"""

    t = t or np.linspace(0, 1, len(r))
    dt = np.gradient(t)
    dt3 = dt[:, None]
    dr = np.gradient(r, axis=0)

    v = dr / dt3
    vmag = np.linalg.norm(v, axis=1)
    s = np.cumsum(vmag * dt)

    si = np.linspace(min(s), max(s), len(s))
    ti = np.interp(si, s, t)

    # interp only does one dimension at a time -
    # this splits them up, interps, stacks them
    # also interp has no axis input, so .T twice
    ri = np.vstack([np.interp(ti, t, rc) for rc in r.T]).T

    return ri, si, ti


def frenet_frame(r, t=None):
    """compute frenet frame for curve r = [x y z], parameterized by t
    note: normal vector is undefined when curvature = 0
    source: thomas calculus e11 ch13
    """
    # TODO: there should be a way to define a normal vector for a straight-line segment
    #       unambiguously in many cases
    #       - if line lies in xy plane, let B = z
    #       - if line is preceded by a curved segment, N, B = N, b at end of curved segment
    #       - isolated straight line in space is ambiguous

    # deal with inputs
    t = t or np.linspace(0, 1, len(r))  # (scalar)

    if r.shape[1] == 2:
        # add third dimension (zeros)
        r = np.hstack((r, np.zeros((len(r), 1))))  # (vector)

    # calculations
    dt = np.gradient(t)                         #          (scalar)
    dt3 = dt[:, None]                           #          (conceptual scalar, numpy vector)
    dr = np.gradient(r, axis=0)                 #          (vector)
    v = dr / dt3                                # velocity (vector)
    dv = np.gradient(v, axis=0)                 #          (vector)
    a = dv / dt3                                # acceleration (vector)
    vmag = np.linalg.norm(v, axis=1)            # speed    (scalar)
    s = np.cumsum(vmag * dt)                    # arclen   (scalar)
    T = v / vmag[:, None]                       # tangent  (vector)
    dT = np.gradient(T, axis=0)                 #          (vector)
    dTdt = dT / dt3                             #          (vector)
    dTdtmag = np.linalg.norm(dTdt, axis=1)      #          (scalar)
    N = dTdt / dTdtmag[:, None]                 # normal   (vector)
    B = np.cross(T, N)                          # binormal (vector)
    dB = np.gradient(B, axis=0)                 #          (vector)
    k = dTdtmag / vmag                          # curvature (scalar)
    dBdotN = np.sum(dB * N, axis=1)             #          (scalar)
    tau = -dBdotN / vmag                        # torsion  (scalar)

    return T, N, B

    
"""discrete derivative must work by computing slope at each point
- first difference: m_i = (y_{i+1} - y_i)/(x_{i+1} - y_i)
slope from point i to point i+1
this is what np.diff does
- better idea: approximate interior points by fitting quadratic
approximate endpoints by fitting linear
this is what np.gradient does
"""
# deprecated
def frenet_frame_difference(r, t=None):
    # deal with inputs
    if not t:
        t = np.linspace(0, 1, len(r))
    t = t[:, None]  # expand axis

    if r.shape[1] == 2:
        # add third dimension (zeros)
        r = np.hstack((r, np.zeros((len(r), 1))))
    D = r.shape[1]

    # calculations
    dt = np.diff(t, axis=0)
    dr = np.diff(r, axis=0)
    v = np.vstack((dr / dt, [np.nan] * D))      # velocity
    dv = np.diff(v, axis=0)
    a = np.vstack((dv / dt, [np.nan] * D))      # acceleration
    vmag = np.linalg.norm(v, axis=1)
    # s = np.cumsum(vmag) ??                    # TODO: arclength
    T = v / vmag[:, None]                       # tangent
    dT = np.diff(T, axis=0)
    dTdt = np.vstack((dT / dt, [np.nan] * D))
    dTdtmag = np.linalg.norm(dTdt, axis=1)
    N = dTdt / dTdtmag[:, None]                 # normal
    B = np.cross(T, N)                          # binormal
    dB = np.diff(B, axis=0)
    k = dTdtmag / vmag                          # curvature
    dBdotN = np.sum(np.vstack((dB, [np.nan] * D)) * N, axis=1)
    tau = -dBdotN / vmag                        # torsion

    return T, N, B

#!/usr/bin/python
import numpy as np


def involute(r):
    pass


def evolute(r):
    pass


def radial(r):
    pass


def parallel_curve(r):
    pass


def add_frenet_offset_2D(base_curve, offset_vector, base_T=None, base_N=None):
    """apply 2D offset_vector using (T, N) of base_curve as the (x, y) plane"""
    # allow inputting T, N because they can be computed more accurately
    # in the caller if they are segments of a longer curve

    if base_T is None or base_N is None:
        base_T, base_N = frenet_frame_2D_corrected(base_curve)

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
    # TODO: use Curve class
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


def frenet_frame_2D(r):
    c = Curve2D(r)
    return c.T, c.N


def frenet_frame_2D_corrected(r):
    c = Curve2D(r)
    c.compute_normal_corrected()
    return c.T, c.N_corrected


class Curve2D(object):
    eps = .00001

    def __init__(self, r, t=None):
        # deal with inputs
        self.t = t if t is not None else np.linspace(0, 1, len(r))  # (scalar)
        self.r = r

        # calculations
        self.dt = np.gradient(self.t)
        self.dt3 = self.dt[:, None]
        self.dr = np.gradient(self.r, axis=0)
        self.v = self.dr / self.dt3
        self.vmag = np.linalg.norm(self.v, axis=1)
        self.T = self.v / self.vmag[:, None]
        self.dT = np.gradient(self.T, axis=0)
        self.dTdt = self.dT / self.dt3
        self.dTdtmag = np.linalg.norm(self.dTdt, axis=1)
        self.N = self.dTdt / self.dTdtmag[:, None]

    def naive_normal(self):
        self.compute_normal_naive()
        return self.N_naive

    def corrected_normal(self):
        self.compute_corrected_normal()
        return self.N_corrected

    def compute_normal_naive(self):
        """compute 2D frenet frame naively.
        normal vector is simple 90 degree rotation of tangent vector"""
        # this means the normal vector does not point into a curve, as expected

        #self.T = self.dr / self.drmag[:, None]
        self.N_naive = np.matrix([[0, 1], [-1, 0]]) * self.T

    def compute_normal_corrected(self):
        """compute frenet frame for a C1 curve r = [x y], parameterized by t.
        N is defined for each value of t, in order of preference:
        - analytical approximation: (dT/dt) / |dT/dt|
        - most recent well-defined value (should work for C1 internal linear segment)
        - a_z cross T (should work for linear segment at start of curve
        """

        # compute N corrected (assuming C1 curve)
        Nlist = []
        last_N = None
        for N, den, tan in zip(self.N, self.dTdtmag, self.T):
            if np.all(abs(den) > self.eps):
                # best case: use proper definition of N
                Nlist.append(N)
                last_N = Nlist[-1]
            elif last_N is not None:
                # use most recent N
                Nlist.append(last_N)
            else:
                # if starting with a line segment, then use a_z cross T
                Nlist.append([-tan[1], tan[0]])

        self.N_corrected = np.vstack(Nlist)


def arc_length_approx(r):
    """compute arc length of line segments of trajectory
    might call this arc_length_first_order, """
    c = Curve(r)
    raise(NotImplementedError)
    # TODO: implement this


def curve_length(r, t=None):
    # this is not defined in Curve because it's a constant, not a
    # function of the parameter
    c = Curve(r, t)
    return np.trapz(c.vmag, c.t)
    

def arc_length(r, t=None):
    """compute arc length assuming the trajectory
    is sampled from a continuous curve"""
    c = Curve(r, t)
    return c.s


def frenet_frame(r, t=None):
    """compute frenet frame for curve r = [x y z], parameterized by t
    note: normal vector is undefined when curvature = 0
    """
    c = Curve(r, t)
    return c.tangent, c.normal, c.binormal

class Curve(object):
    """numeric representation of space curve
    source: thomas calculus e11 ch13"""

    def __init__(self, r=None, t=None):
        # different init types:
        # Curve(r) - define trajectory with uniform spacing in [0, 1]
        # Curve(r, t)
        # Curve(, tau, kappa) - define from torsion and curvature 
        # TODO: split up into an input handle init, and multiple calculators
        # below is the Curve(r) and Curve(r, t) version (linear basis)

        # deal with inputs
        self.r = r
        self.t = t or np.linspace(0, 1, len(r))  # (scalar)

        if r.shape[1] == 2:
            # add third dimension (zeros)
            self.r = np.hstack((r, np.zeros((len(r), 1))))  # (vector)

        # calculations
        # TODO: rename these properly, but then also make the nicknames work
        self.dt = np.gradient(self.t)                         #          (scalar)
        self.dt3 = self.dt[:, None]                           #          (conceptual scalar, numpy vector)
        self.dr = np.gradient(self.r, axis=0)                 #          (vector)
        self.v = self.dr / self.dt3                                # velocity (vector)
        self.dv = np.gradient(self.v, axis=0)                 #          (vector)
        self.a = self.dv / self.dt3                                # acceleration (vector)
        self.vmag = np.linalg.norm(self.v, axis=1)            # speed    (scalar)
        # TODO: use cumtrapz somehow
        #self.s = np.cumsum(self.vmag * self.dt)                    # arclen   (scalar) 
        self.T = self.v / self.vmag[:, None]                       # tangent  (vector)
        self.dT = np.gradient(self.T, axis=0)                 #          (vector)
        self.dTdt = self.dT / self.dt3                             #          (vector)
        # TODO: fix warning?
        # import ipdb; ipdb.set_trace()
        self.dTdtmag = np.linalg.norm(self.dTdt, axis=1)      #          (scalar)
        self.N = self.dTdt / self.dTdtmag[:, None]                 # normal   (vector)
        self.B = np.cross(self.T, self.N)                          # binormal (vector)
        self.dB = np.gradient(self.B, axis=0)                 #          (vector)
        self.k = self.dTdtmag / self.vmag                          # curvature (scalar)
        self.dBdotN = np.sum(self.dB * self.N, axis=1)             #          (scalar)
        self.tau = -self.dBdotN / self.vmag                        # torsion  (scalar)



def pseudo_frenet_frame(r, t=None):
    # TODO: just define this in Curve class
    """compute pseudo-frenet frame (still works for straight line segments)
    pseudo-frenet frame is an entity that looks similar to a frenet frame,
    but is well-defined for a straight line, thanks to an arbitrary but
    consistent choice of binormal direction."""
    # this depends on the arbitrary choice of a pseudo-binormal _B, which
    # is set to the unit z vector here. This has benefits:
    # - if line lies in x-y plane, then B = a_z (thus N in x-y plane)
    # - if line does not lie in xy plane, then B is the vector "closest" to a_z in the
    #   line's normal plane
    # if T too close to a_z, then just choose _B = a_y
    #
    # TODO: expand this function to also work for curves, plus:
    # - use (N, B) of the previous segment, for a line segment preceeded by a curve
    # - do similar on other end of line segment
    # - smoothly interpolate from (N_0, B_0) at start to (N_1, B_1) at end

    _B = np.array([0, 0, 1])

    dt = np.gradient(t)
    dt3 = dt[:, None]
    dr = np.gradient(r, axis=0)
    v = dr / dt3
    vmag = np.linalg.norm(v, axis=1)
    T = v / vmag[:, None]
    if T[0, 2] > 0.99:
        # TODO: verify this works
        _B = np.array([0, 1, 0])

    Ndir = np.cross(_B, T)
    N = Ndir / np.linalg.norm(Ndir, axis=1)
    B = np.cross(T, N)

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

import numpy as np
from curves import frenet_frame


def extrude(f=None, g=None):
    """Extrude a planar curve along a space curve
    g = [gx(u,v) gy(u,v)]    - planar curve with additional param
    f = [fx(u) fy(u) fz(u)]  - space curve

    returns parametric surface [x(u, v), y(u, v), z(u, v)]

    f should be an (Nu x 3) array
    g should be a (3 x Nu x Nv) array
    Sx, Sy, Sz are (Nu x Nv) arrays

    when no inputs are supplied, uses a trefoil space curve and a circle cross section
    """

    # both f and g can be defined as fourier series
    # which makes variation of g along f easy to define

    if f is None and g is None:
        # use a trefoil curve and a circle
        uvec, vvec = np.linspace(0, 2*np.pi, 64), np.linspace(0, 2*np.pi, 64)
        ugrid, vgrid = np.meshgrid(uvec, vvec)
        f = np.stack([2*np.sin(2*uvec)-np.sin(uvec),
                      2*np.cos(2*uvec)+np.cos(uvec),
                      2*np.sin(3*uvec)]).T

        g = np.stack((0.25*np.cos(vgrid),
                      0.25*np.sin(vgrid),
                      0*vgrid))

    T, N, B = frenet_frame(f)
    Sx = f[:, 0] + g[0, :, :]*N[:, 0] + g[1, :, :]*B[:, 0] + g[2, :, :]*T[:, 0]
    Sy = f[:, 1] + g[0, :, :]*N[:, 1] + g[1, :, :]*B[:, 1] + g[2, :, :]*T[:, 1]
    Sz = f[:, 2] + g[0, :, :]*N[:, 2] + g[1, :, :]*B[:, 2] + g[2, :, :]*T[:, 2]
    return Sx, Sy, Sz



# raymarching surface-point distance functions
# primitives are centered at origin, axis aligned, etc. inverse transform point, to transform object
# http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

def sphere_point_distance(p, r):
    # sphere of radius r at origin
    # point p
    return np.linalg.norm(p) - r


def box_point_distance():
    pass


def rounded_box_point_distance():
    pass


def torus_point_distance(p, t1, t2):
    # torus with... ?
    # vec2 q = vec2(length(p.xz)-t.x,p.y);
    # return length(q)-t.y;
    q = np.array([np.linalg.norm(p.xz)-t1, p.y])
    return np.linalg.norm(q) - t2


def cylinder_point_distance(p, c):
    # cylinder with...?
    pass


def capped_cylinder_point_distance(p, c):
    pass


def cone_point_distance(p, c):
    # cone with...?
    pass

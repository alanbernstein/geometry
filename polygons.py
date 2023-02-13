import numpy as np
from line import Line


def polyarea(poly):
    # shoelace formula AKA surveyor's formula
    # works for convex or nonconvex, but not self-intersecting
    x, y = zip(*poly)
    yp = y[1:] + y[:1]
    yn = y[-1:-1] + y[:-1]
    return abs([xi * (yip-yin) for xi, yip, yin in zip(x, yp, yn)])/2.0


def in_polygon(xq, yq, poly):
    """
    test whether (xq, yq) is inside poly
    https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon/2922778#2922778
int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
{
  int i, j, c = 0;
  for (i = 0, j = nvert-1; i < nvert; j = i++) {
    if ( ((verty[i]>testy) != (verty[j]>testy)) &&
     (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
       c = !c;
  }
  return c;
}

    """
    x, y = zip(*poly)

    i, j, c = 0, len(x)-1, False
    while i < len(x):
        # must compute in floating point
        if ((y[i] > yq) != (y[j] > yq) and
            (xq < (x[j]-x[i]) * (yq-y[i]) / (1.0*y[j]-y[i]) + x[i])):
            c = not c

        i, j = i+1, i

    return c


def clip_polygon(subject, clip):
    # https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm

    clipped = subject

    return clipped


def get_polygon_convexity_ccw(pts):
    # input:  Nx2 array
    # output: [is_convex, is_convex, ...] x N
    # only works for ccw polygons (cross(v0, v1) < 0)
    # but same approach will work for both if just determine whether ccw or not
    i0 = np.arange(len(pts))
    iprev = (i0 - 1) % len(pts)
    inext = (i0 + 1) % len(pts)
    v0 = pts[inext] - pts[i0]
    v1 = pts[i0] - pts[iprev]
    return np.cross(v0, v1) < 0


def regular_polygon(sides=3):
    return np.exp(np.arange(sides) * 2j * np.pi / sides)


def random_convex_polygon(sides=3, debias=True):
    angles = np.arange(0, 1, 1./sides)
    noise = (np.random.random(sides)-0.5)/sides
    angles = (angles + noise) * 2 * np.pi
    # angles = np.random.random(sides) * 2 * np.pi
    angles.sort()
    v = np.exp(1j*angles)
    if debias:
        v -= np.mean(v)
    return v


def regular_star(p=5, q=2, r=1.0):
    # see e.g. http://mathworld.wolfram.com/StarPolygon.html
    # in that context,
    # p = number of points
    # q = "density" - how many vertices are skipped. 1 < q < p/2
    # -
    # r = distance from origin to star point

    # 1. this works for q=2
    r_in = r * np.cos(np.pi * q/p)  # inradius of innermost polygon
    r_mid = r_in / np.cos(np.pi/p)  # circumradius of innermost polygon
    # r_mid = r * np.cos(np.pi * q/p) / np.cos(np.pi/p)

    # 2. works for any q, but brute forcely. oh well
    P = r * np.exp(2.0j * np.pi * np.array([0, q, 1, 1-q])/p)
    l0 = Line(p1=[P[0].real, P[0].imag], p2=[P[1].real, P[1].imag])
    l1 = Line(p1=[P[2].real, P[2].imag], p2=[P[3].real, P[3].imag])
    in_pt = l0.intersect(l1)
    r_mid = np.hypot(in_pt[0], in_pt[1])

    # 3. works for any q, simple
    star_internal_angle = np.pi * (p-2*q)/p
    polygon_internal_angle = np.pi * (p-2)/p
    triangle_angle = (polygon_internal_angle - star_internal_angle)/2
    # triangle_angle = np.pi * (q-1.0) / p
    # https://en.wikipedia.org/wiki/Regular_polygon#Circumradius
    side_midpoint_dist = r * np.cos(np.pi/p)  # apothem of p-gon
    triangle_base = r * 2 * np.sin(np.pi/p)   # side of p-gon
    triangle_height = np.tan(triangle_angle) * triangle_base/2
    r_mid = side_midpoint_dist - triangle_height

    # 3.1 simplified all the way:
    # TODO: why does this agree with 1 when q=2?
    r_mid = r * (np.cos(np.pi/p) - np.tan(np.pi * (q-1.0) / p) * np.sin(np.pi/p))

    # simple after computing the midradius:
    n = np.arange(0., 2*p)
    angles = n / p * np.pi
    mags = r_mid + (r - r_mid) * (n % 2)
    v = mags * np.exp(1j*angles)
    return v

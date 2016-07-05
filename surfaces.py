from curves import frenet_frame


def extrude(f, g, uvec, vvec):
    """Extrude a planar curve along a space curve
    g = [gx(u,v) gy(u,v)]    - planar curve with additional param
    f = [fx(u) fy(u) fz(u)]  - space curve

    plane curve g is a fourier series as a function of both u and v - easy to
    define the cross section as any shape, also easy to define the variation
    along the space curve

    space curve is also defined as a fourier series (or will be eventually) -
    default definition gives trefoil curve

    returns parametric surface [x(u, v), y(u, v), z(u, v)]"""
    pass

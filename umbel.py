import numpy as np

from shapes import arc


def umbel_logo():
    # shape     pixels        relative
    # circle D           90   1.5
    # bars width         64   1
    # middle bar length  195  3
    # side bar length    120  2
    # bottom circle D         5
    # circle-bar dist    40   .666
    # bottom circles concentric
    paths = [
        np.vstack((
            arc(2.5j, 2.5, np.pi, 2*np.pi),   # main U - bottom arc outside
            arc(2+4.5j, .5, 0, np.pi),         # main U - top-right arc
            arc(2.5j, 1.5, 2*np.pi, np.pi),  # main U - bottom arc inside
            arc(-2+4.5j, .5, 0, np.pi),        # main U - top-left arc
            [-2.5, 2.5],
        )),
        arc(2+(5.7+2.0/3)*1j, .7),         # right circle
        arc((6.7+2.0/3)*1j, .7),            # middle circle
        arc(-2+(5.7+2.0/3)*1j, .7),         # left circle
        np.vstack((
            arc(2.5j, 0.5, np.pi, 2*np.pi),  # middle bar - bottom arc
            arc(5.5j, 0.5, 0, np.pi),        # middle bar - top arc
            [-0.5, 2.5],
        ))
    ]

    # not as useful for the shape-packing thing...
    # l = [[p, []] for p in paths]
    # mp = MultiPolygon(l)

    PLOT = False
    if PLOT:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        for p in paths:
            plt.plot(p[:, 0], p[:, 1])
        plt.show()

    return paths
    # return mp


if __name__ == '__main__':
    from shapely.geometry import MultiPolygon
    paths = umbel_logo()
    l = [[p, []] for p in paths]
    mp = MultiPolygon(l)

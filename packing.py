#!/usr/local/bin/python
"""
appolonian gasket, but with umbel logo pieces
https://bl.ocks.org/mbostock/7607535
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from descartes.patch import PolygonPatch

from umbel import umbel_logo
from logos import pilosa_logo
from shapes import arc, square
from ipdb import iex, set_trace as db
from svgpathtools import svg2paths

from svglib.svglib import svg2rlg

# try using something like http://cimar.mae.ufl.edu/CIMAR/pages/thesis/Pasha_A_CISE.pdf
# https://github.com/Jack000/SVGnest

def read_svg(img_fname, dt=1/100.):
    # read SVG line drawing from file
    # return a list of paths, each of which is a list of complex numbers x+iy
    # also return a list of list of segments, each a list of complex numbers
    # [[[c0, c1, c2, ...], [...], ...], ...]
    #   segment
    #  path
    # paths
    paths, attributes = svg2paths(img_fname)
    pths = []
    paths_seg = []
    for n, path in enumerate(paths):
        print(n, len(path))
        pth = []
        pth_segs = []
        for segment in path:
            # print(segment, segment.point(0), segment.point(1))
            seg = []
            for t in np.arange(0, 1, dt):
                pth.append(complex(segment.point(t)))
                seg.append(complex(segment.point(t)))
            pth_segs.append(seg)
        paths_seg.append(pth_segs)
        pths.append(pth)
    return pths, paths_seg


@iex
def main():
    # umbel = umbel_logo()

    poop_pths, poop_segs = read_svg('poop-emoji.svg', dt=1/20)
    poop_pths[0] = poop_pths[0][0:560]

    poop_paths = []
    for pth in poop_pths:
        x = [k.real for k in pth]
        y = [-k.imag for k in pth]
        path = np.vstack((x, y)).T
        poop_paths.append(path)

    """
    plt.figure()
    for pth in poop_pths:
        x = [k.real for k in pth]
        y = [-k.imag for k in pth]
        plt.plot(x, y)
    plt.axis('equal')
    plt.show()
    db()
    """

    dim = np.array([6, 4])

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.xlim([0, dim[0]])
    plt.ylim([0, dim[1]])
    plt.grid()

    plt.get_current_fig_manager().window.raise_()

    # sp = ShapePacker(dim, [arc()])
    # sp = ShapePacker(dim, [umbel_logo()])
    sp = ShapePacker(dim, [poop_paths])
    sp.pack(100, ax)


class ShapePacker(object):
    def __init__(self, dim=None, base_shapes=None, boundaries=None):
        if dim is None:
            dim = np.array([6, 4])
        self.dim = dim

        if base_shapes is None:
            base_shapes = [arc()]
        if boundaries is None:
            boundaries = [x for x in base_shapes]
        self.base_shapes = base_shapes
        self.boundaries = boundaries

        border_int = square() * dim
        border_ext = square() * (dim + 1) - [0.5, 0.5]
        self.border_poly = Polygon(border_ext, [border_int])

    def plot_border(self):
        plot_polys([self.border_poly])

    def pack(self, count=3, ax=None):
        # if ax present, update plot with every new poly
        self.shapes = [self.border_poly]
        if ax:
            self.plot_border()
        for n in range(count):
            print(n)
            self.pack_shape()
            if ax:
                plot_polys(self.shapes[-1:], ax)
                # plt.show(block=False)
                # plt.pause(.1)

        plt.show()
        db()

    def random_interior_point(self):
        # get point inside
        return np.random.random((1, 2)) * self.dim

    def random_point(self, use_boundaries=True):
        # get point inside interior and not inside another poly
        count = 0
        while True:
            count += 1
            c = self.random_interior_point()
            pt = Point(c[0, 0], c[0, 1])
            intersected = False
            for shape in self.shapes:
                if shape.contains(pt):
                    intersected = True
                    break
            if not intersected:
                break
        # print('  random point, %d tries' % count)
        return c

    def pack_shape(self, *args, **kwargs):
        return self.pack_shape_scale(*args, **kwargs)

    def pack_shape_scale_position_rotation(self, plot=False):
        # 
        pass

    def pack_shape_scale_position(self, plot=False):
        pass

    def pack_shape_scale(self, plot=False):
        # TODO: select one of the base_shapes randomly
        # TODO: use bounding circle for first pass
        #       https://www.nayuki.io/res/smallest-enclosing-circle/smallestenclosingcircle.py
        center = self.random_point()
        base = self.base_shapes[0]
        ph = np.random.random() * 2 * np.pi
        R = np.matrix([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
        rbase = [b * R for b in base]

        if plot:
            self.plot_border()
            plt.plot(center[:, 0], center[:, 1], 'k.')
            hc, = plt.plot([], [], 'k-')

        # binary search on scale to find best fit
        thresh = .001
        lo = 0
        r = .001
        hi = np.inf
        while True:
            transformed = [[(r * b + center).tolist(), []] for b in rbase]

            # a = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
            # b = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]
            # mp = MultiPolygon([[a, []], [b, []]])
            p = MultiPolygon(transformed)
            intersected = False
            for shape in self.shapes:
                if p.intersects(shape):
                    intersected = True
                    break

            if intersected:
                # too big, reduce size and adjust range
                hi = r
                r = (lo + r) / 2
            else:
                # too small, increase size
                lo = r
                if hi == np.inf:
                    # keep doubling while we haven't yet hit a neighboring shape
                    r = 2*r
                else:
                    r = (r + hi) / 2

            if plot:
                cc = arc(center, r)
                hc.set_xdata(cc[:, 0])
                hc.set_ydata(cc[:, 1])
                plt.show(block=False)
                # plt.pause(.01)

            if hi - lo <= thresh:
                break

        # print('  radius = %f' % r)

        self.shapes.append(p)

    def pack_shape_scale_linear(self):
        center = self.random_point()
        base = self.base_shapes[0]
        ph = np.random.random() * 2 * np.pi
        R = np.matrix([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
        rbase = base * R

        # linear search on scale to find best fit
        r = 0
        delta = 2 ** -4
        while True:
            p = Polygon(r * rbase + center)
            intersected = False
            for shape in self.shapes:
                for poly in shape:
                    if p.intersects(poly):
                        intersected = True
                        break
                if intersected:
                    break

            # if any([p.intersects(poly) for poly in polys]):
            if intersected:
                break
            r += delta
        print('  %f' % r)

        self.shapes.append(p)


def plot_polys(shapes, ax=None):
    if not ax:
        ax = plt.gca()
    for shape in shapes:
        if type(shape) == Polygon:
            polys = [shape]
        else:
            polys = shape
        for p in polys:
            patch = PolygonPatch(p, facecolor='None', edgecolor='k', alpha=0.5, zorder=2)
            ax.add_patch(patch)


if __name__ == '__main__':
    main()

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from mpltools import unpack_plot_kwargs
# implement a vector font for use in laser designs

# lower case letters:
# y=0 baseline
# centered horizontally
# n-width = 1

# https://mzucker.github.io/2016/08/03/miniray.html
# https://github.com/cmiscm/leonsans/blob/master/src/font/lower.js
# https://github.com/cmiscm/leonsans

h = .75  # height of an "h" extender, above the base "o" circle
d = .75  # depth of a "q" descender, below the base "o" circle


class Path(object):
    def plot(self, offset=None, radius=0, italic_angle=0, **plot_kwargs):
        # offset: translation vector
        # radius: extrude a circle of this radius along the path (requires combination of distance functions?)
        # TODO: thickness via radius (bold)
        # TODO: italics via shear
        offset = offset or np.array([0, 0])
        plot_kw = unpack_plot_kwargs(plot_kwargs)
        plt.plot(offset[0] + self.x, offset[1] + self.y, **plot_kw)

    def plot_aux_lines(self):
        # used by plot_debug in child
        plt.plot([-1, 1], [0, 0], '-.', color='gray')
        plt.plot([-1, 1], [1, 1], '-.', color='gray')
        plt.plot([-1, 1], [1+h, 1+h], '-.', color='gray')
        plt.plot([-1, 1], [-d, -d], '-.', color='gray')


class Line(Path):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.x = np.array([p1[0], p2[0]])
        self.y = np.array([p1[1], p2[1]])

    def plot_debug(self, **plot_kwargs):
        plt.plot(self.x, self.y, 'k.')
        self.plot_aux_lines()
        self.plot(**plot_kwargs)


class Arc(Path):
    def __init__(self, c, r, a1=None, a2=None):
        # center, radius, angle start, angle end (both in fractions of pi)
        self.c = c
        self.r = r
        self.a1 = a1 or 0
        self.a2 = a2 or 2
        circ = self.r*np.exp(np.pi*1j*np.linspace(self.a1, self.a2, 65))
        self.x = circ.real + self.c[0]
        self.y = circ.imag + self.c[1]

    def plot_debug(self, **plot_kwargs):
        if self.a1 != 0 or self.a2 != 2:
            circ = self.r*np.exp(np.pi*1j*np.linspace(0, 2, 65))
            plt.plot(circ.real + self.c[0], circ.imag + self.c[1], '--', color='gray')
        plt.plot(self.c[0], self.c[1], 'k.')
        self.plot_aux_lines()
        self.plot(**plot_kwargs)


a2 = -.25  # the beginning angle of the arc on the 2 TODO: ensure arc-line joint is continuous slope

chars = {
    # character: list of lines/arcs
    ' ': [],
    '0': [Arc([0, 0.5], 0.5)],
    '1': [Line([-.5, 0], [.5, 0]), Line([0, 0], [0, 1+h]), Line([0, 1+h], [-h/2, 1+h/2])],
    '2': [Arc([0, .5+h], .5, a2, 1), Line([.5*np.cos(a2*np.pi), .5+h+.5*np.sin(a2*np.pi)], [-.5, 0]), Line([-.5, 0], [.5, 0])],
    '3': [Line([-.5, 1+h], [.5, 1+h]), Line([.5, 1+h], [0, 1]), Arc([0, .5], .5, -1, .5)],
    '4': [Line([.5, 0], [.5, 1+h]), Line([.5, 1+h], [-.5, .5]), Line([-.5, .5], [.5, .5])],
    '5': [Line([.5, 1+h], [-.5, 1+h]), Line([-.5, 1+h], [-.5, 1]), Line([-.5, 1], [0, 1]), Arc([0, .5], .5, -1, .5)],
    '6': [Line([-.5, .5], [-.5, 1+h-.5]), Arc([0, .5], .5), Arc([0, 1+h-.5], .5, 0, 1)],
    '7': [Line([-.5, 0], [.5, 1+h]), Line([.5, 1+h], [-.5, 1+h])],
    '8': [Arc([0, .5], .5), Arc([0, 1+h/2], h/2)],
    '9': [Line([.5, .5], [.5, 1+h-.5]), Arc([0, 1+h-.5], .5), Arc([0, .5], .5, -1, 0.01)],  # TODO: why doesnt a2=0 work?
    'a': [Line([0.5, 0], [0.5, 1]), Arc([0, 0.5], 0.5)],
    'b': [Line([-0.5, 0], [-0.5, 1+h]), Arc([0, 0.5], 0.5)],
    'c': [Arc([0, 0.5], 0.5, 1/4, 7/4)],
    'd': [Line([0.5, 0], [0.5, 1+h]), Arc([0, 0.5], 0.5)],
    'e': [Line([-0.5, 0.5], [0.5, 0.5]), Arc([0, 0.5], 0.5, 0, 7/4)],
    'f': [Arc([0, 1+h-.25], .25, .25, 1), Line([-.25, 1+h-.25], [-.25, 0]), Line([-.5, 1], [0, 1])],
    'g': [Line([0.5, 1], [0.5, 0]), Arc([0, 0.5], 0.5), Arc([0, 0], .5, 1, 2)],
    'h': [Line([-.5, 0], [-.5, 1+h]), Line([.5, 0], [.5, .5]), Arc([0, 0.5], .5, 0, 1)],
    'i': [Line([0, 0], [0, 1]), Arc([0, 1.25], 1/16)],
    'j': [Line([0, 0], [0, 1]), Arc([-.25, 0], .25, 1, 2), Arc([0, 1.25], 1/16)],
    'k': [Line([-.5, 0], [-.5, 1+h]), Line([-.5, .5], [.25, 0]), Line([-.5, .5], [.25, 1])],
    'l': [Line([0, .25], [0, 1+h]), Arc([.25, .25], .25, 1, 1.5)],
    'm': [Line([-.5, 0], [-.5, 1]), Arc([-.25, .75], .25, 0, 1), Arc([.25, .75], .25, 0, 1), Line([.5, 0], [.5, .75]), Line([0, 0], [0, .75])],
    'n': [Line([-.5, 0], [-.5, 1]), Arc([0, 0.5], .5, 0, 1), Line([.5, 0], [.5, .5])],
    'o': [Arc([0, 0.5], 0.5)],
    'p': [Line([-0.5, 1], [-0.5, -d]), Arc([0, 0.5], 0.5)],
    'q': [Line([0.5, 1], [0.5, -d]), Arc([0, 0.5], 0.5)],
    # 'r': [Line([-.5, 0], [-.5, 1]), Arc([np.sqrt(2)/4-.5, .5], .5, .25, .75)],
    'r': [Line([-.5, 0], [-.5, 1]), Arc([0, .5], .5, .25, 1)],
    # 's': [Arc([0, .25], .25, -.5, .5), Arc([0, .75], .25, .5, 1.5), Line([-.25, 0], [0, 0]), Line([0, 1], [.25, 1])],
    # 's': [Arc([0, .25], .25, -.75, .5), Arc([0, .75], .25, .25, 1.5)],
    's': [Arc([0, .25], .25, -1, .5), Arc([0, .75], .25, 0, 1.5)],
    't': [Line([0, .25], [0, 1.25]), Arc([.25, .25], .25, 1, 1.5), Line([-.25, 1], [.25, 1])],
    'u': [Line([.5, 0], [.5, 1]), Arc([0, 0.5], .5, 1, 2), Line([-.5, .5], [-.5, 1])],
    'v': [Line([-.5, 1], [0, 0]), Line([0, 0], [.5, 1])],
    'w': [Line([-.5, 1], [-.25, 0]), Line([-.25, 0], [0, .75]), Line([0, .75], [.25, 0]), Line([.25, 0], [.5, 1])],
    'x': [Line([-0.5, 1], [0.5, 0]), Line([-0.5, 0], [0.5, 1])],
    'y': [Line([-.5, 1], [0, 0]), Line([-.375, -.75], [.5, 1])],
    'z': [Line([-.5, 1], [.5, 1]), Line([.5, 1], [-.5, 0]), Line([-.5, 0], [.5, 0])],
    '°': [Arc([0, 1+h-.25], .25)],
    '.': [Arc([0, 1/16], 1/16)],
    '-': [Line([-.25, .5], [.25, .5])],
    '--': [Line([-.5, .5], [.5, .5])],  # em dash
    '_': [Line([-.5, 0], [.5, 0])],
    '~': [],
    '|': [Line([0, 0], [0, 1+h])],
    '?': [],
    '!': [],
    '@': [],
    '#': [],
    '$': [],
    '%': [],
    '^': [],
    '&': [],
    '*': [],
    '(': [],
    ')': [],
    '[': [],
    ']': [],
    '+': [],
    '/': [],
    '\\': [],  # single backslash
    '<': [],
    '>': [],
    ',': [],
}


kern = 0.25
kerns = defaultdict()


def plot_string(s, offset=None):
    offset = offset or [0, 0]
    for n, c in enumerate(s):
        for part in chars[c]:
            # TODO: bounding-box kerning
            # TODO: pairwise custom kerning
            part.plot(offset=[offset[0] + n*(1+kern), offset[1]], color='k', linewidth=1)


def string_to_paths(s, scale=1.0, offset=None):
    # TODO each letter an svg group
    offset = offset or [0, 0]
    pths = []
    for n, c in enumerate(s):
        for part in chars[c]:
            x = offset[0] + scale*(n*(1+kern) + part.x)
            y = offset[1] + scale*(part.y)
            pths.append(np.vstack((x, y)).T)
    return pths


def test_all():
    # test plot all characters
    plt.figure()
    x = 3
    plot_string('abcdefghi', [0, 0])
    plot_string('jklmnopqr', [0, -1*x])
    plot_string('stuvwxyz', [0, -2*x])
    plot_string('0123456789', [0, -3*x])
    plot_string('3.4°5-6_7|8#', [0, -4*x])
    plt.axis('equal')
    plt.grid(False)


def test_angle_labels():
    plt.figure()
    plot_string('4.0°   3.9°   3.8°', [0, 0])
    plt.axis('equal')
    plt.grid(False)


def test_single_character(c):
    # examine a single character
    plt.figure()
    for part in chars[c]:
        part.plot_debug(color='k')
    plt.axis('equal')

if __name__ == '__main__':
    test_all()
    test_single_character('s')
    plt.show()

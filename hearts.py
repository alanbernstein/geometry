#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from line import Line

from ipdb import iex, set_trace as db

# NOTE: rethink all of this with a broader guideline born from:
# - creating a constant-width outline strip: do it with an SVG stroke, separately
#   - but for a heart, want the two vertex joints to be pointy on the sharp side
# - beveling/chamfering: not sure if i should implement or outsource somehow

figsize = (10, 10)
dpi = 80

@iex
def demo():
    demo_overview_grid()
    #demo_overlaid()
    # demo_parameter_scale()

    #h_construct(100, 45)
    #h_analytical(100, 45)

    #demo_equal_vertex_angles()
    #demo_90_bottom()
    #demo_90_top()
    #demo_supplementary_vertex_angles()

    plt.show()

def demo_overview_grid():
    plt.figure()
    for n in range(9):
        plt.subplot(3, 3, n+1)
        hearts[n]().plot()

def demo_overlaid():
    # 1x1 superimposed all
    plt.figure()


def demo_equal_vertex_angles():
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('equal')
    for a in [70, 80, 90, 100, 110]:
        path = h(a, a)
        plt.plot(path.real, path.imag, label='A=B=%d' % a)

    plt.legend()
    plt.grid('off')
    plt.axis('off')
    plt.title('Same vertex angle on top and bottom')
    plt.savefig('heart-demo1.png')


def demo_supplementary_vertex_angles():
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('equal')
    for d in [-20, -10, 0, 10, 20]:
        path = h(90+d, 90-d)
        plt.plot(path.real, path.imag, label='A=%d, B=%d' % (90+d, 90-d))

    plt.legend()
    plt.grid('off')
    plt.axis('off')
    plt.title('Supplementary vertex angles on top and bottom')
    plt.savefig('heart-demo2.png')



def demo_90_bottom():
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('equal')
    # bs = [0, 45, 90, 135, 180]
    bs = [45, 67, 90, 112, 135]
    for b in bs:
        path = h(90, b)
        plt.plot(path.real, path.imag, label='B=%d' % b)

    #plt.text(0.1, 0, 'A')
    #plt.text(0, 1.1, 'B')
    plt.grid('off')
    plt.axis('off')
    plt.legend()
    plt.title('Bottom=90, vary top angle')
    plt.savefig('heart-demo3.png')


def demo_90_top():
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('equal')
    # bs = [0, 45, 90, 135, 180]
    aa = [45, 67, 90, 112, 135]
    for a in aa:
        path = h(a, 90)
        plt.plot(path.real, path.imag, label='A=%d' % a)

    #plt.text(0.1, 0, 'A')
    #plt.text(0, 1.1, 'B')
    plt.grid('off')
    plt.axis('off')
    plt.legend()
    plt.title('Top=90, vary bottom angle')
    plt.savefig('heart-demo4.png')


def h_analytical(a_degrees=90, b_degrees=84):
    a = a_degrees * np.pi/180
    b = b_degrees * np.pi/180
    a2, b2 = a/2, b/2

    r = 1/((np.cos(b2)+np.cos(a2))/np.tan(a2) + np.sin(b2)+np.sin(a2))
    w = r * (np.cos(b2) + np.cos(a2))
    h = w / np.tan(a2)
    cr = w-r*np.cos(a2) + 1j*(h + r*np.sin(a2))
    cl = -np.conj(cr)

    ar = cr + r*np.exp(1j*np.linspace(-a2, np.pi-b2, 32))
    al = cl + r*np.exp(1j*np.linspace(b2, np.pi+a2, 32))

    path = np.hstack((
        0,
        ar,
        al[1:],
        0,
    ))

    PLOT = True
    if PLOT:
        plt.plot(path.real, path.imag, 'y--')
    return path


def h_construct(alpha_degrees=90, beta_degrees=84):
    # type: custom-parametric
    # description: heart with two circular-arc lobes,
    # connected via tangent lines that meet at the bottom vertex.
    #
    # alpha = full angle of bottom interior (acute)
    # beta = full angle of top exterior (acute)
    N = 64
    nn = np.linspace(0, N+1)
    cc = np.exp(2j*np.pi*nn/N)

    alpha = alpha_degrees * np.pi/180
    beta = beta_degrees * np.pi/180
    a = np.pi/2 - (alpha)/2            # angle from x-axis to bottom direction
    b = np.pi/2 - (beta)/2             # angle from x-axis to top direction

    l1 = Line(p1=[0, 0], angle=a)          # line through bottom vertex
    l2 = Line(p1=[0, 1], angle=b)          # line through top vertex

    # need to compute center (i2) and radius (r) of circles that form
    # lobes of heart
    if a == b:
        # special case: parallel vectors
        d = np.cos(a)  # distance between parallel aux lines
        r = d/2        # circle radius
        i2 = [0+r*np.cos(a-np.pi/2), 1+r*np.sin(a-np.pi/2)]  # circle center

        # auxiliary objects for demo plot, for parity with the a != b case
        c = a
        i1 = [np.nan, np.nan]  # point at infinity (don't plot)
        i3 = [0+d*np.cos(a-np.pi/2), 1+d*np.sin(a-np.pi/2)]
    else:
        i1 = l1.intersect(l2)                  # auxiliary point
        c = (a+b)/2                            # direction angle of angle bisector
        l3 = Line(p1=i1, angle=c)              # angle bisector
        l4 = Line(p1=[0, 1], angle=b-np.pi/2)  # upper radius line
        i2 = l3.intersect(l4)                  # circle center
        r = np.sqrt(i2[0]**2 + (i2[1]-1)**2)   # circle radius
        l5 = Line(p1=i2, angle=a-np.pi/2)      # lower radius line
        i3 = l5.intersect(l1)                  # bottom circle tangent point

    # need i2, r
    phi0 = a-np.pi/2
    phi1 = b+np.pi/2
    ar = (i2[0]+1j*i2[1]) + r*np.exp(1j*np.linspace(phi0, phi1, 32))
    phi2 = np.pi/2-b
    phi3 = 3*np.pi/2-a
    al = (-i2[0]+1j*i2[1]) + r*np.exp(1j*np.linspace(phi2, phi3, 32))

    path = np.hstack((
        0,
        ar,
        al[1:],
        0,
    ))

    PLOT = True
    if PLOT:
        ak = {'head_width': 0.025, 'head_length': 0.05}
        lk = {'linewidth': 1, 'linestyle': '--'}

        for stage in range(7, 8):
            plt.figure(figsize=figsize, dpi=dpi)
            if stage >= 0:
                plt.title('Inputs')
                plt.plot(0, 0, 'ro')
                plt.plot(0, 1, 'bo')
                plt.arrow(0, 0, np.cos(a)/4, np.sin(a)/4, color='r', **ak)
                plt.arrow(0, 1, np.cos(b)/4, np.sin(b)/4, color='b', **ak)

            if stage >= 1:
                plt.title('Extend lines, find intersection')
                plt.plot([0, i1[0]], [0, i1[1]], 'r--', **lk)
                plt.plot([0, -i1[0]], [0, -i1[1]], 'r--', **lk)
                plt.plot([0, i1[0]], [1, i1[1]], 'b--', **lk)
                plt.plot(i1[0], i1[1], 'm.')

            if stage >= 2:
                plt.title('Angle bisector')
                plt.arrow(i1[0], i1[1], np.cos(c)/4, np.sin(c)/4, color='m', **ak)
                plt.plot([i1[0], i2[0]], [i1[1], i2[1]], 'm--', **lk)

            if stage >= 3:
                plt.title('Circle center')
                plt.arrow(0, 1, np.cos(b-np.pi/2)/4, np.sin(b-np.pi/2)/4, color='g', **ak)
                plt.plot([0, i2[0]], [1, i2[1]], 'g--', **lk)
                plt.plot(i2[0], i2[1], 'k.')

            if stage >= 4:
                plt.title('Auxiliary circle')
                plt.plot(i2[0]+r*cc.real, i2[1]+r*cc.imag, 'gray', **lk)

            if stage >= 5:
                plt.title('Lower radius')
                plt.plot(i3[0], i3[1], 'c.')
                plt.plot([i2[0], i3[0]], [i2[1], i3[1]], 'c', **lk)
                plt.arrow(i2[0], i2[1], np.cos(a-np.pi/2)/4, np.sin(a-np.pi/2)/4, color='c', **ak)

            if stage >= 6:
                plt.title('Reflect circle')
                plt.plot(-i2[0], i2[1], 'k.')
                plt.plot(-i2[0]+r*cc.real, i2[1]+r*cc.imag, 'gray', **lk)
                plt.plot(-i3[0], i3[1], 'c.')

            if stage >= 7:
                plt.title('Connect the two arcs and two lines')
                plt.plot(path.real, path.imag, 'k-')

                

            plt.axis('equal')
            plt.grid('off')
            plt.axis('off')
            #plt.savefig('heart-stage-%d.png' % stage)

        #plt.figure(figsize=figsize, dpi=dpi)
        #plt.title('Final shape')
        #plt.plot(path.real, path.imag, 'k-')
        #plt.axis('equal')
        #plt.grid('off')
        #plt.axis('off')
        #plt.savefig('heart-final.png')
        #plt.show()

    return path


h = h_analytical

def hthick(alpha_degrees, beta_degrees, width=0.125):
    # TODO
    pass


# https://mathworld.wolfram.com/HeartCurve.html
# h0 - h7 follow the ordinal descriptions on the mathworld page
# h8 is from a reddit post
class Heart(object):
    def plot(self):
        title = '%s - %s' % (self.__class__.__name__, self.name)
        print(title)

        if self.typ == 'implicit':
            plt.contour(self.X, self.Y, self.F, [0], colors=[(1, 0, 0)])
        elif self.typ == 'polar':
            plt.plot(self.x, self.y, 'k')
        elif self.typ == 'parametric':
            plt.plot(self.x, self.y, 'b')

        plt.axis('equal')
        plt.title(title)

class H0(Heart):
    typ = 'polar'
    name = 'cardioid'
    def __init__(self):
        t = np.linspace(0, 2*np.pi, 256+1)
        r = 1 - np.sin(t)
        self.x, self.y = np.cos(t)*r, np.sin(t)*r

class H1(Heart):
    typ = 'implicit'
    name = 'heart surface cross-section'
    def __init__(self):
        W = 2.0
        delta = 0.01
        x = np.arange(-W, W, delta)
        X, Y = np.meshgrid(x, x)
        self.X, self.Y = X, Y
        self.F = (X*X + Y*Y - 1) ** 3 - (X*X*Y*Y*Y)

class H2(Heart):
    typ = 'parametric'
    name = 'Dascanio 2003'
    def __init__(self):
        t = np.linspace(-1, 1, 256+1)
        t = t ** 5
        self.x = np.sin(t)*np.cos(t)*np.log(np.abs(t))
        self.y = np.abs(t)**0.3 * np.sqrt(np.cos(t))

class H3(Heart):
    typ = 'implicit'
    name = 'Kuriscak 2006'
    def __init__(self):
        W = 8.0
        delta = 0.01
        x = np.arange(-W, W, delta)
        X, Y = np.meshgrid(x, x)
        self.X, self.Y = X, Y
        self.F = X**2 + (Y - (2*(X*X+np.abs(X)-6))/(3*(X*X+np.abs(X)+2)))**2-36

class H4(Heart):
    typ = 'polar'
    name = 'anonymous wolfram alpha user'
    def __init__(self):
        t = np.linspace(0, 2*np.pi, 256+1)
        r = np.sin(t) * np.sqrt(np.abs(np.cos(t))) / (np.sin(t) + 7/5) - 2*np.sin(t) + 2
        self.x, self.y = np.cos(t)*r, np.sin(t)*r

class H5(Heart):
    typ = 'parametric'
    name = '?'
    def __init__(self):
        t = np.linspace(0, 2*np.pi, 256+1)
        self.x = 16*np.sin(t)**3
        self.y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

class H6(Heart):
    typ = 'implicit'
    name = 'Schroeder 2021'
    def __init__(self):
        W = 2.0
        delta = 0.01
        x = np.arange(-W, W, delta)
        X, Y = np.meshgrid(x, x)
        self.X, self.Y = X, Y
        self.F = (Y - (X * X)**(1/3)) ** 2 + X**2 - 1

class H7(Heart):
    typ = 'parametric'
    name = 'nephroid, Mangaldan 2023'
    def __init__(self):
        t = np.linspace(0, 2*np.pi, 256+1)
        self.x = -np.sqrt(2)*np.sin(t)**3
        self.y = -np.cos(t)**3 - np.cos(t)**2 + 2*np.cos(t)

class H8(Heart):
    typ = 'implicit'
    name = 'HSzold 2023'
    # https://old.reddit.com/r/math/comments/111wv99/i_sent_my_girlfriend_this_mathematical_poem_for/
    def __init__(self):
        W = 2.0
        delta = 0.01
        x = np.arange(-W, W, delta)
        X, Y = np.meshgrid(x, x)
        self.X, self.Y = X, Y
        self.F = X*X + Y*Y - np.abs(X)*Y - 1


hearts = [H0, H1, H2, H3, H4, H5, H6, H7, H8]

# join two half-capsules [crossing angle]

if __name__ == "__main__":
    demo()

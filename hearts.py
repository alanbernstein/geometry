#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from line import Line

from ipdb import iex, set_trace as db

# NOTE: rethink all of this with a broader guideline born from:
# - creating a constant-width outline strip: do it with an SVG stroke, separately
#   - but for a heart, want the two vertex joints to be pointy on the sharp side
# - beveling/chamfering: not sure if i should implement or outsource somehow


@iex
def demo():
    #demo_overview_grid()
    #demo_overlaid()
    # demo_parameter_scale()
    demo_equal_vertex_angles()
    demo_90_bottom()
    plt.show()

def demo_overview_grid():
    # 3x3 subplot grid of heart types
    plt.figure()

def demo_overlaid():
    # 1x1 superimposed all
    plt.figure()


def demo_equal_vertex_angles():
    plt.figure()
    plt.axis('equal')
    for a in [70, 80, 90, 100, 110]:
        path = h1(a, a)
        plt.plot(path.real, path.imag, label='A=B=%d' % a)

    plt.legend()
    plt.suptitle('same vertex angle on top and bottom')

def demo_90_bottom():
    plt.figure()
    plt.axis('equal')
    # bs = [0, 45, 90, 135, 180]
    bs = [45, 67, 90, 112, 135]
    for b in bs:
        path = h1(90, b)
        plt.plot(path.real, path.imag, label='B=%d' % b)

    plt.text(0.1, 0, 'A')
    plt.text(0, 1.1, 'B')
    plt.legend()
    plt.suptitle('A=90, vary B')


# type = [polar, parametric, implicit, custom]

# join two half-capsules [crossing angle]
# join two ellipses


def h1(alpha_degrees, beta_degrees):
    # type: custom-parametric
    # description: heart with two circular-arc lobes,
    # connected via tangent lines that meet at the bottom vertex.
    # 
    # alpha = full angle of bottom interior (acute)
    # beta = full angle of top exterior (acute)
    N = 64
    nn = np.linspace(0, N+1)
    cc = np.exp(2j*np.pi*nn/N)

    a = np.pi/2 - (alpha_degrees * np.pi/180)/2  # angle from x-axis to bottom direction
    b = np.pi/2 - (beta_degrees * np.pi/180)/2   # angle from x-axis to top direction

    l1 = Line(p1=[0, 0], angle=a)          # line through bottom vertex
    l2 = Line(p1=[0, 1], angle=b)          # line through top vertex

    # need to compute center (i2) and radius (r) of circles that form
    # lobes of heart
    if a == b:
        d = np.cos(a)  # distance between parallel aux lines
        r = d/2
        i2 = [0+r*np.cos(a-np.pi/2), 1+r*np.sin(a-np.pi/2)]

        # auxiliary objects for demo plot, for parity with the a != b case
        c = a
        i1 = [np.nan, np.nan]  # point at infinity
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

    PLOT = False
    if PLOT:
        ak = {'head_width': 0.025, 'head_length': 0.05}
        lk = {'linewidth': 1, 'linestyle': '--'}

        plt.plot(0, 0, 'ro')
        plt.plot(0, 1, 'bo')
        plt.arrow(0, 0, np.cos(a)/4, np.sin(a)/4, color='r', **ak)
        plt.arrow(0, 1, np.cos(b)/4, np.sin(b)/4, color='b', **ak)

        plt.plot([0, i1[0]], [0, i1[1]], 'r--', **lk)
        plt.plot([0, -i1[0]], [0, -i1[1]], 'r--', **lk)
        plt.plot([0, i1[0]], [1, i1[1]], 'b--', **lk)

        plt.plot(i1[0], i1[1], 'm.')
        plt.arrow(i1[0], i1[1], np.cos(c)/4, np.sin(c)/4, color='m', **ak)
        plt.plot([i1[0], i2[0]], [i1[1], i2[1]], 'm--', **lk)

        plt.arrow(0, 1, np.cos(b-np.pi/2)/4, np.sin(b-np.pi/2)/4, color='g', **ak)
        plt.plot([0, i2[0]], [1, i2[1]], 'g--', **lk)

        plt.plot(i2[0], i2[1], 'k.')
        plt.plot(i2[0]+r*cc.real, i2[1]+r*cc.imag, 'gray', **lk)

        plt.plot(i3[0], i3[1], 'c.')
        plt.plot([i2[0], i3[0]], [i2[1], i3[1]], 'c', **lk)
        plt.arrow(i2[0], i2[1], np.cos(a-np.pi/2)/4, np.sin(a-np.pi/2)/4, color='c', **ak)

        plt.plot(-i2[0], i2[1], 'k.')
        plt.plot(-i2[0]+r*cc.real, i2[1]+r*cc.imag, 'gray', **lk)
        plt.plot(-i3[0], i3[1], 'c.')

        plt.plot(path.real, path.imag, 'k-')

        plt.axis('equal')

    return path



    
# https://mathworld.wolfram.com/HeartCurve.html
# r = 1 sin(?)
# ?
# [np.sin(t)*np.cos(t)*np.log(np.abs(t)), np.abs(t)**0.3 * np.sqrt(np.cos(t))]
# implicit
# r=
# []
# implicit
    
    
if __name__ == "__main__":
    demo()

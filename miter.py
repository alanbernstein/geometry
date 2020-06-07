import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mpltools import equal_box
from geometry.polyhedra import select_shortest_edges, cube
from geometry.plane import Plane
from debug import pm, debug


@pm
def main():
    miter_45_45()


def miter_45_45():
    W_2x4 = 3.5
    H_2x4 = 1.5
    L_top = 6
    # top board

    cube_dict = cube()
    vc = np.array(cube_dict['vertices'])
    ec = cube_dict['edges']

    e1 = ec

    Sx, Sy, Sz = L_top/2, W_2x4/2, H_2x4/2
    v1_a = vc @ np.diag([Sx, Sy, Sz])  # scale cube to 2x4 shape
    v1_b = v1_a
    # manually define deviation from that shape
    v1_b[0,:] -= [W_2x4, 0, 0]
    v1_b[1,:] -= [W_2x4, 0, 0]
    v1_b[0,:] -= [H_2x4, 0, 0]
    v1_b[3,:] -= [H_2x4, 0, 0]

    # rotate 45 degrees to axis-align the bevel angle
    c, s = np.sqrt(2)/2, np.sqrt(2)/2
    v1_c = v1_b @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    v1 = v1_c


    # find bevel angle
    v1_2d = v1[[0, 1, 5, 4, 0], :][:,[0, 2]]
    opp, adj = v1_2d[1, 0] - v1_2d[0, 0], v1_2d[1, 1] - v1_2d[0, 1]
    # opp, adj = 1.5, 1.5/sqrt(2)
    # opp, adj = 1, 1/r2
    # opp, adj = r2, 1
    # so this is the acute angle of a 1-sqrt(2)-sqrt(3) right triangle
    miter_bevel_angle = np.arctan(opp/adj)
    print('bevel angle: %f rad = %f deg' % (miter_bevel_angle, miter_bevel_angle * 180/np.pi))
    # bevel angle: 0.615480 rad = 35.264390 deg


    # dihedral angle between planes defined by vertices [0, 3, 2] and [7, 3, 0]
    p1 = Plane(p1=v1[0,:], p2=v1[3,:], p3=v1[2,:])
    p2 = Plane(p1=v1[7,:], p2=v1[3,:], p3=v1[0,:])
    angle = p1.dihedral_angle(p2)
    print('dihedral angle: %f rad = %f deg' %  (angle, angle*180/np.pi))
    # dihedral angle: 1.932163 rad = 110.704811 deg
    # not useful for miter saw cuts


    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for e in e1:
        plt.plot(*zip(*[v1[z] for z in e]), color='k', linewidth=3)

    plt.plot(*zip(*v1), color='y', marker='o', linestyle='None', markersize=6)

    for n, x in enumerate(v1):
        plt.gca().text(x[0], x[1], x[2], '%d' % n, horizontalalignment='center', verticalalignment='center', color='k')

    ax.plot(**equal_box(5))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.figure()
    plt.plot(v1_2d[:,0], v1_2d[:,1])
    plt.plot([v1_2d[0,0], v1_2d[1, 0], v1_2d[1, 0]], [v1_2d[0,1], v1_2d[1, 1], v1_2d[0, 1]], 'r-')


    plt.axis('equal')

    plt.show()

    debug()


main()


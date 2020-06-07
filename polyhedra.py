import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

from panda.debug import debug, pm

# coordinate definitions


def tetrahedron():
    vertices = [
        [-1, -1, -1],
        [1, 1, -1],
        [-1, 1, 1],
        [1, -1, 1]
    ]
    edges = select_shortest_edges(vertices)
    return {'vertices': vertices, 'edges': edges}


def cube():
    vertices = [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [1, 1, -1],
    ]
    edges = select_shortest_edges(vertices)
    return {'vertices': vertices, 'edges': edges}


hexahedron = cube


def octahedron():
    vertices = [
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0],
    ]
    edges = select_shortest_edges(vertices)
    return {'vertices': vertices, 'edges': edges}


p = (np.sqrt(5) + 1)/2
pi = 1 / p


def dodecahedron():
    vertices = [
        [0, pi, p],
        [0, pi, -p],
        [0, -pi, p],
        [0, -pi, -p],
        [p, 0, pi],
        [p, 0, -pi],
        [-p, 0, pi],
        [-p, 0, -pi],
        [pi, p, 0],
        [pi, -p, 0],
        [-pi, p, 0],
        [-pi, -p, 0],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]
    edges = select_shortest_edges(vertices)
    return {'vertices': vertices, 'edges': edges}


def icosahedron():

    vertices = [
        [0, p, 1],
        [0, p, -1],
        [0, -p, 1],
        [0, -p, -1],
        [1, 0, p],
        [1, 0, -p],
        [-1, 0, p],
        [-1, 0, -p],
        [p, 1, 0],
        [p, -1, 0],
        [-p, 1, 0],
        [-p, -1, 0],
    ]
    edges = select_shortest_edges(vertices)
    return {'vertices': vertices, 'edges': edges}


icosahedron_circumradius = None
truncated_icosahedron_circumradius = np.sqrt(9*p+10)


def truncated_icosahedron():
    vertices = [
        [0, 1, 3*p],
        [0, 1, -3*p],
        [0, -1, 3*p],
        [0, -1, -3*p],

        [1, 2+p, 2*p],
        [1, 2+p, -2*p],
        [1, -2-p, 2*p],
        [1, -2-p, -2*p],
        [-1, 2+p, 2*p],
        [-1, 2+p, -2*p],
        [-1, -2-p, 2*p],
        [-1, -2-p, -2*p],

        [p, 2, 2*p+1],
        [p, 2, -2*p-1],
        [p, -2, 2*p+1],
        [p, -2, -2*p-1],
        [-p, 2, 2*p+1],
        [-p, 2, -2*p-1],
        [-p, -2, 2*p+1],
        [-p, -2, -2*p-1],
    ]
    # generate all even permutations of each of ^
    v1 = [[x[2], x[0], x[1]] for x in vertices]
    v2 = [[x[1], x[2], x[0]] for x in vertices]
    vertices = vertices + v1 + v2

    edges = select_shortest_edges(vertices)
    return {'vertices': vertices, 'edges': edges}


def icosahedron_cap(upright=True):
    p = icosahedron()
    # v_base = 0
    # ei = [0, 1, 2, 3, 4, 7, 8, 18, 19, 24]
    # edges = [p['edges'][i] for i in ei]
    # e_select = [e for e in p['edges'] if v_base in e]

    vi = [0, 1, 4, 6, 8, 10]
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 4],
        [4, 2],
        [2, 3],
        [3, 5],
        [5, 1]
    ]
    vertices = [p['vertices'][i] for i in vi]

    if not upright:
        return {'vertices': vertices, 'edges': edges}

    # rotate from v_from to v_to
    v_from = vertices[0] / np.linalg.norm(vertices[0])
    v_to = [0, 0, 1]
    dot = np.dot(v_from, v_to)
    cross = np.cross(v_from, v_to)
    cross_norm = cross/np.linalg.norm(cross)
    R = Rotation.from_rotvec(cross_norm*np.arccos(dot))
    v_rotated = R.apply(vertices)

    return {'vertices': v_rotated, 'edges': edges}


def cylinder(num_sides=6, height=1):
    vertices = []
    edges = []
    for n in range(num_sides):
        a = 2 * np.pi * n / num_sides
        vertices.append([np.cos(a), np.sin(a), 0])
        vertices.append([np.cos(a), np.sin(a), height])
        edges.append([2*n, 2*n+1])   # 2n, 2n+1
        edges.append([2*n, 2 * ((n+1) % num_sides)])  # 2n, 2n+2
        edges.append([2 * ((n+1) % num_sides) - 1, 2 * ((n+1) % num_sides) + 1])  # 2n+1, 2n+3

    return {'vertices': vertices, 'edges': edges}


def pyramid(num_sides=6, height=1):
    vertices = []
    edges = []
    for n in range(num_sides):
        a = 2 * np.pi * n / num_sides
        vertices.append([np.cos(a), np.sin(a), 0])
        edges.append([n, (n+1) % num_sides])
        edges.append([n, num_sides])

    vertices.append([0, 0, height])
    return {'vertices': vertices, 'edges': edges}


def bipyramid(num_sides=6, height=1):
    vertices = []
    edges = []
    for n in range(num_sides):
        a = 2 * np.pi * n / num_sides
        vertices.append([np.cos(a), np.sin(a), 0])
        edges.append([n, (n+1) % num_sides])
        edges.append([n, num_sides])
        edges.append([n, num_sides+1])

    vertices.extend([[0, 0, height], [0, 0, -height]])

    return {'vertices': vertices, 'edges': edges}


# computational geometry utilities

def select_shortest_edges(vertices):
    def dist2(x, y):
        return sum([float(a-b)**2 for a, b in zip(x, y)])

    # compute all edge lengths, and the min length
    N = len(vertices)
    all_edges = []
    min_d = 10000
    for n1 in range(N-1):
        for n2 in range(n1+1, N):
            d = dist2(vertices[n1], vertices[n2])
            all_edges.append((n1, n2, d))
            if d < min_d:
                min_d = d

    # select all edges that are floating-point-equal to the min
    thresh = min_d + 0.000001
    valid_edges = [e[0:2] for e in all_edges if e[2] <= thresh]
    return valid_edges


def compute_regular_faces(vertices, edges, face_count=None):
    # convex hull? alpha shape?
    # for special case of archimedean/platonic solids:
    # - identify all sets of coplanar points
    # - sort those sets by the distance-from-centroid-to-origin
    # - select the N faces with largest distances
    #   - N can be provided, or can stop when total steradians subtended reaches 4pi
    # - orient the faces by sorting by area around centroid-origin vector
    # - check that each edge in this candidate face is present in the known edges list

    v = np.array(vertices)
    # compute list of all sets of coplanar points, along with the distance-to-origin of the plane they define
    planes_dict = {}  # use dict to avoid duplicate planes
    for i0 in range(0, len(v)-2):
        # TODO for v[i2], only choose among those vertices which are connected to v[i1] with an edge
        # this loop takes 3sec for truncated icosahedron (60 vertices), but pretty fast for anything else
        for i1 in range(i0+1, len(v)-1):
            # TODO similarly for v2/v3
            for i2 in range(i1+1, len(v)):
                # compute plane parameters a,b,c,d from points v[i0], v[i1], v[i2]
                a = (v[i1][1] - v[i0][1]) * (v[i2][2] - v[i0][2]) - (v[i2][1] - v[i0][1]) * (v[i1][2] - v[i0][2])
                b = (v[i1][2] - v[i0][2]) * (v[i2][0] - v[i0][0]) - (v[i2][2] - v[i0][2]) * (v[i1][0] - v[i0][0])
                c = (v[i1][0] - v[i0][0]) * (v[i2][1] - v[i0][1]) - (v[i2][0] - v[i0][0]) * (v[i1][1] - v[i0][1])
                d = -(a*v[i1][0] + b*v[i1][1] + c*v[i1][2])

                # compute distance to plane for all vertices, threshold
                D = (a*v[:, 0] + b*v[:, 1] + c*v[:, 2] + d)/np.sqrt(a*a+b*b+c*c)
                vis, = np.where(np.abs(D) < .001)

                # append the list of vertexes, and distance to plane, to the face list
                centroid = np.mean(v[vis, :], axis=0)

                planes_dict[tuple(vis.tolist())] = np.linalg.norm(centroid)

    planes = [(k, v) for k, v in planes_dict.items()]
    planes.sort(key=lambda x: -x[1])
    faces_oriented = []
    n = 0
    total_solid_angle = 0
    # while len(faces_oriented) < face_count:
    # add faces in decreasing order of distance from origin,
    # subject to the constraint that they be composed of valid edges,
    # and stop when the total solid angle subtended by them all reaches
    # 4pi, the solid angle of the complete sphere
    while total_solid_angle < 4*np.pi - .00001:
        f = planes[n][0]
        vf = v[f, :]
        centroid = np.mean(vf, axis=0)

        if np.linalg.norm(centroid) == 0:
            # if centroid = 0, can't possibly be a face
            n += 1
            continue

        # rotate all points such that their centroid is on the z-axis
        v_from = centroid / np.linalg.norm(centroid)
        v_to = [0, 0, 1]
        dot = np.dot(v_from, v_to)
        cross = np.cross(v_from, v_to)
        if np.linalg.norm(cross) != 0:
            cross_norm = cross/np.linalg.norm(cross)
            R = Rotation.from_rotvec(cross_norm*np.arccos(dot))
            vf_rotated = R.apply(vf)
        else:
            # face is already aligned to the z-axis
            # TODO: may need to fix anti-aligned orientation
            vf_rotated = vf

        # sort by angle
        angle = np.arctan2(vf_rotated[:, 1], vf_rotated[:, 0])
        idx = np.argsort(angle)
        f_sorted = [f[i] for i in idx]

        # check that each edge is valid
        valid = True
        for k in range(len(f_sorted)):
            k1 = (k+1) % len(f_sorted)
            e = tuple(sorted([f_sorted[k], f_sorted[k1]]))
            if e not in edges:
                valid = False
                break

        # if all valid, found a face
        if valid:
            faces_oriented.append(f_sorted)
            total_solid_angle += polygon_solid_angle(v[f_sorted, :])

        n += 1

    return faces_oriented


# rendering functions
def plot_polyhedron_wireframe(vertices=None, edges=None, **kwargs):

    # ax.plot(*zip(*vertices), color='r', marker='o', linestyle='None')
    # equals this:
    # for v in vertices:
    #     ax.plot([v[0]], [v[1]], [v[2]], 'ko')

    for e in edges:
        plt.plot(*zip(*[vertices[z] for z in e]), **kwargs)
        # equals this:
        # ax.plot([vertices[e[0]][0], vertices[e[1]][0]],
        #         [vertices[e[0]][1], vertices[e[1]][1]],
        #         [vertices[e[0]][2], vertices[e[1]][2]], 'k-')


def plot_polyhedron_solid(ax, vertices=None, edges=None, faces=None, color='g', alpha=1):
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.Polygon.html
    v = np.array(vertices)
    face_coords = [v[f, :] for f in faces]
    face_object = a3.art3d.Poly3DCollection(face_coords)
    # face_object.set_color(colors.rgb2hex([.2, 1, .2]))
    face_object.set_alpha(alpha)  # NOTE: call set_alpha BEFORE set_color or it won't work
    face_object.set_color(color)
    face_object.set_edgecolor('k')
    ax.add_collection3d(face_object)


def animate_truncated_icosahedron():
    # TODO: animate all the way through the dodecahderon
    # then run
    # convert -delay 13 frames/*.png -loop 0 trunc.gif
    # to compile to gif

    # plt.style.use('dark_background')

    icosa = icosahedron()
    icosa_f = compute_regular_faces(icosa['vertices'], icosa['edges'])
    v1 = np.array(icosa['vertices'])
    R = Rotation.from_rotvec([0, 0, np.pi/2])
    v1 = 3*R.apply(v1)
    m1 = np.linalg.norm(np.mean(v1[icosa_f[0], :], axis=0))  # inradius

    trunc = truncated_icosahedron()
    trunc_f = compute_regular_faces(trunc['vertices'], trunc['edges'])
    v2 = np.array(trunc['vertices'])
    m2 = np.linalg.norm(np.mean(v2[trunc_f[0], :], axis=0))  # two options here

    dodeca = dodecahedron()
    dodeca_f = compute_regular_faces(dodeca['vertices'], dodeca['edges'])
    v3 = np.array(dodeca['vertices'])
    R1 = Rotation.from_rotvec([0, 0, np.pi/2])
    m3 = np.linalg.norm(np.mean(v3[dodeca_f[0], :], axis=0))
    scale = (9-np.sqrt(5))/2  # m2/m3 equals this (inverse symbolic calculator)
    v3 = scale * R1.apply(v3)

    nearest_idxs_1 = []
    nearest_idxs_3 = []
    for v in v2:
        nearest_idxs_1.append(np.argmin(np.linalg.norm(v-v1, axis=1)))
        nearest_idxs_3.append(np.argmin(np.linalg.norm(v-v3, axis=1)))

    if True:
        # single test frame - truncated icosahedron -> dodecahedron
        S = 1
        v4 = []
        for va, ib in zip(v2, nearest_idxs_3):
            vb = v3[ib]
            vc = va + S*(vb-va)
            v4.append(vc)

        fig = plt.figure(figsize=plt.figaspect(1)*2)
        ax = fig.add_subplot(111, projection='3d')
        # plt.axis('off')
        ax.auto_scale_xyz(1, 1, 1)
        # plot_polyhedron_wireframe(v2, trunc['edges'], color='b', linestyle='-')
        plot_polyhedron_wireframe(v4, trunc['edges'], color='r', linestyle='-')
        # plot_polyhedron_solid(ax, v1, icosa['edges'], icosa_f, color='gray', alpha=0)
        # plot_polyhedron_solid(ax, v4, trunc['edges'], trunc_f, color='g', alpha=1)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        ax.set_zlim3d([-4, 4])
        plt.show()
        return

    if True:
        # single test frame - truncated icosahedron -> icosahedron
        S = 0
        v4 = []
        for va, ib in zip(v2, nearest_idxs_1):
            vb = v1[ib]
            vc = va + S*(vb-va)
            v4.append(vc)

        fig = plt.figure(figsize=plt.figaspect(1)*2)
        ax = fig.add_subplot(111, projection='3d')
        plt.axis('off')
        ax.auto_scale_xyz(1, 1, 1)
        # plot_polyhedron_wireframe(v1, icosa['edges'], color='w', linestyle='-')
        # plot_polyhedron_solid(ax, v1, icosa['edges'], icosa_f, color='gray', alpha=0)
        plot_polyhedron_solid(ax, v4, trunc['edges'], trunc_f, color='g', alpha=1)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        ax.set_zlim3d([-4, 4])
        plt.show()
        return

    # loop over frame index
    t_vec = np.linspace(0, 2*np.pi, 60)
    Smin, Smax = -.5, 1
    S = -.5  # parameter to control adjustment toward/away nearest v1 vertex
    for n, t in enumerate(t_vec):
        S = (Smax-Smin)*(np.sin(t)+1)/2 + Smin
        v4 = []
        for va, ib in zip(v2, nearest_idxs_1):
            vb = v1[ib]
            vc = va + S*(vb-va)
            v4.append(vc)

        fig = plt.figure(figsize=plt.figaspect(1)*2)
        ax = fig.add_subplot(111, projection='3d')
        # plot_animation_frame(ax, v1, icosa['edges'], v3, trunc['edges'], trunc_f)
        plot_polyhedron_solid(ax, v4, trunc['edges'], trunc_f, color='g', alpha=1)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        ax.set_zlim3d([-4, 4])
        plt.axis('off')
        # plt.savefig('frames/trunc-anim-%02d.png' % n, bbox_inches='tight')
        plt.savefig('frames/trunc-anim-%02d.png' % n)
        plt.close()
        print(n)

    # plt.show()


def polygon_solid_angle(v):
    # v is a list of vertices, equidistant from the origin, ordered in correct sequence as described above
    # break polygon into triangles
    # compute spherical excess E = A + B + C - pi
    # area = radius^2 * E  (girard theorem)
    # subtended solid angle = area (when radius = 1)
    # subtended angle of full polygon = sum of angles of component triangles (gauss-bonnet?)

    angle = 0
    for n in range(1, len(v) - 1):
        # https://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
        # L'Huilier's formula - spherical equivalent of hero's formula
        A, B, C = v[0], v[n], v[n+1]
        c = np.arccos(np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B)))
        a = np.arccos(np.dot(B, C)/(np.linalg.norm(B)*np.linalg.norm(C)))
        b = np.arccos(np.dot(C, A)/(np.linalg.norm(C)*np.linalg.norm(A)))
        s = (a + b + c)/2
        E = 4 * np.arctan(np.sqrt(np.tan(s/2) * np.tan((s-a)/2) * np.tan((s-b)/2) * np.tan((s-c)/2)))
        angle += E

    return angle


def plot_animation_frame(ax, v1, e1, v2, e2, f2):
    plot_polyhedron_wireframe(v1, e1, color='w', linestyle='-')
    plot_polyhedron_solid(ax, v2, e2, f2, color='g')


def test_plot_wireframe():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot_polyhedron_wireframe(**cube())
    # plot_polyhedron_wireframe(**dodecahedron())
    # plot_polyhedron_wireframe(color='r', **icosahedron_cap())
    # plot_polyhedron_wireframe(**icosahedron_cap(upright=False))
    plot_polyhedron_wireframe(color='r', **truncated_icosahedron())

    # plot_polyhedron_wireframe(**cylinder())
    # plot_polyhedron_wireframe(**pyramid(num_sides=64))
    plt.show()


def test_plot_solid():
    fig = plt.figure()

    stuff = [
        (tetrahedron, 4),
        (cube, 6),
        (octahedron, 8),
        (dodecahedron, 12),
        (icosahedron, 20),
        (truncated_icosahedron, 32),
    ]

    for n, data in enumerate(stuff):
        poly_func, face_count = data
        ax = fig.add_subplot(231+n, projection='3d')
        poly = poly_func()
        v, e = poly['vertices'], poly['edges']
        t0 = time.time()
        faces = compute_regular_faces(v, e)
        t1 = time.time()
        print('%f sec to find faces for %s' % (t1-t0, poly_func.__name__))
        plot_polyhedron_solid(ax, v, e, faces)
        ax.set_xlim([-4, 4]), ax.set_ylim([-4, 4]), ax.set_zlim([-4, 4])

    plt.show()


def test_solid_angle():
    stuff = [
        (tetrahedron, 4),
        (cube, 6),
        (octahedron, 8),
        (dodecahedron, 12),
        (icosahedron, 20),
        (truncated_icosahedron, 32),
    ]
    for n, data in enumerate(stuff):
        poly_func, face_count = data
        poly = poly_func()
        v, e = np.array(poly['vertices']), poly['edges']
        faces = compute_regular_faces(v, e)
        angles = [polygon_solid_angle(v[f, :]) for f in faces]
        print(n, poly_func.__name__, sum(angles))


@pm
def main():
    # test_plot_wireframe()
    # test_plot_solid()
    # test_solid_angle()
    animate_truncated_icosahedron()


if __name__ == '__main__':
    main()

import math


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


p = (math.sqrt(5) + 1)/2
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


def cylinder(num_sides=6, height=1):
    vertices = []
    edges = []
    for n in range(num_sides):
        a = 2 * math.pi * n / num_sides
        vertices.append([math.cos(a), math.sin(a), 0])
        vertices.append([math.cos(a), math.sin(a), height])
        edges.append([2*n, 2*n+1])   # 2n, 2n+1
        edges.append([2*n, 2 * ((n+1) % num_sides)])  # 2n, 2n+2
        edges.append([2 * ((n+1) % num_sides) - 1, 2 * ((n+1) % num_sides) + 1])  # 2n+1, 2n+3

    return {'vertices': vertices, 'edges': edges}


def pyramid(num_sides=6, height=1):
    vertices = []
    edges = []
    for n in range(num_sides):
        a = 2 * math.pi * n / num_sides
        vertices.append([math.cos(a), math.sin(a), 0])
        edges.append([n, (n+1) % num_sides])
        edges.append([n, num_sides])

    vertices.append([0, 0, height])
    return {'vertices': vertices, 'edges': edges}


def bipyramid(num_sides=6, height=1):
    vertices = []
    edges = []
    for n in range(num_sides):
        a = 2 * math.pi * n / num_sides
        vertices.append([math.cos(a), math.sin(a), 0])
        edges.append([n, (n+1) % num_sides])
        edges.append([n, num_sides])
        edges.append([n, num_sides+1])

    vertices.extend([[0, 0, height], [0, 0, -height]])

    return {'vertices': vertices, 'edges': edges}


def plot_polyhedron(vertices, edges):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(*zip(*vertices), color='r', marker='o', linestyle='None')
    # equals this:
    # for v in vertices:
    #     ax.plot([v[0]], [v[1]], [v[2]], 'ko')

    for e in edges:
        ax.plot(*zip(*[vertices[z] for z in e]), color='k', marker='o')
        # equals this:
        # ax.plot([vertices[e[0]][0], vertices[e[1]][0]],
        #         [vertices[e[0]][1], vertices[e[1]][1]],
        #         [vertices[e[0]][2], vertices[e[1]][2]], 'k-')

    plt.show()


if __name__ == '__main__':
    # plot_polyhedron(**cube())
    # plot_polyhedron(**dodecahedron())
    # plot_polyhedron(**icosahedron())
    # plot_polyhedron(**cylinder())
    plot_polyhedron(**pyramid(num_sides=64))

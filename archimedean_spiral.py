import matplotlib.pyplot as plt
import numpy as np


def demo():
    plt.figure()
    path=aspiral()
    plt.axis('equal')
    plt.show()

def aspiral():

    max_diameter= 40  # inches
    Nturns, segments_per_turn = 5, 64

    a = max_diameter / (2 * Nturns * 2*np.pi)
    t = np.arange(0, Nturns * 2*np.pi, 2*np.pi / segments_per_turn)  # theta
    r = a * t  # radius
    x, y = r * np.cos(t), r * np.sin(t)  # cartesian
    k = 2 + t**2 / (a * (1 + t ** 2) ** 3/2)  # curvature
    s = a/2 * (t * np.sqrt(1 + t ** 2) + np.arcsinh(t))  # arc length

    PLOT=True
    if PLOT:
        plt.subplot(211)
        plt.plot(x, y)
        plt.axis('equal')
        plt.title('spiral shape (inches)')

        plt.subplot(212)
        plt.plot(t, s/12)
        plt.xlabel('theta')
        plt.ylabel('arc length (feet)')

    return x+1j*y


if __name__ == '__main__':
    demo()

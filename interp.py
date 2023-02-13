import numpy as np
import scipy.special as sc

def test_slerp():
    import matplotlib.pyplot as plt
    plt.figure()
    x = np.linspace(0, 1, 100)
    plt.plot(x, slerp(x, p=-1), 'y-')
    plt.plot(x, slerp(x, p=0), 'm-')
    plt.plot(x, slerp(x, p=1), 'k-')
    plt.plot(x, slerp(x, p=1.5), 'c-')
    plt.plot(x, slerp(x, p=2), 'r-')
    plt.plot(x, slerp(x, p=3), 'g-')
    plt.plot(x, slerp(x, p=4), 'b-')
    plt.show()

def slerp(x, p=1):
    # y=slerp(x,p) has the following properties:
    # x = 0.0 -> y = 0.0
    # x = 0.5 -> y = 0.5
    # x = 1.0 -> y = 1.0
    # for p >= 1,
    # x = 0.0 -> y' = 0.0
    # x = 0.5 -> y' > 1.0, and y'(.5) is an increasing function of p
    # x = 1.0 -> y' = 0.0
    # as p increases, y(x) becomes flatter at x=0 and x=1
    # p=1 corresponds to y(x) = 3x^2 - 2x^3

    # fully general:
    # y = x ** (P+1) * hypergeom([-P, P+1],[2+P],x)/hypergeom([-P, P+1],[2+P],1)

    f = lambda x: sc.hyp2f1(-p, p+1, 2+p, x)
    y = x ** (p+1) * f(x)/f(1)
    # y = 3*x**2 - 2*x**3
    y[np.where(x<0)] = 0
    y[np.where(x>1)] = 1
    return y

def slerp_func(xmin, xmax, ymin, ymax):
    def f(x):
        return slerp((x-xmin)/(xmax-xmin), 1)*(ymax-ymin) + ymin
    return f


if __name__ == '__main__':
    test_slerp()

import numpy as np

class Line(object):
    """
    A simple line class to compute intersections
    """
    def __init__(self, **kwargs):
        if 'p1' in kwargs and 'p2' in kwargs:
            self._init_point_point(**kwargs)
        elif 'a' in kwargs and 'b' in kwargs and 'c' in kwargs:
            self._init_standard(**kwargs)
        elif 'p1' in kwargs and 'm' in kwargs:
            self._init_point_slope(**kwargs)
        elif 'p1' in kwargs and 'angle' in kwargs:
            self._init_point_angle(**kwargs)
            """
        TODO:
        elif 'm' in kwargs and 'b' in kwargs:
            self._init_slope_intercept(**kwargs)
        elif 'x' in kwargs or 'y' in kwargs:
            self._init_axis_aligned(**kwargs)
            """
        else:
            raise Exception('Line format not supported')

    def __repr__(self):
        return '<Line %s x %+s y = %s>' % (self.a, self.b, self.c)

    def _init_point_slope(self, **kwargs):
        self.p1 = kwargs['p1']
        m = kwargs['m']
        if m == np.inf:
            self.p2 = [self.p1[0], self.p1[1] + 1]
        else:
            self.p2 = [self.p1[0] + 1, self.p1[1] + m]
        self.a = self.p1[1] - self.p2[1]
        self.b = self.p2[0] - self.p1[0]
        self.c = (self.p2[0] - self.p1[0])*self.p1[1] - (self.p2[1]-self.p1[1])*self.p1[0]

    def _init_point_angle(self, **kwargs):
        p1 = kwargs['p1']
        angle = np.array(kwargs['angle'])
        p2 = np.array(p1) + [np.cos(angle), np.sin(angle)]
        self._init_point_point(p1=p1, p2=p2)

    def _init_point_point(self, **kwargs):
        """
        y-y1 = (y2-y1)/(x2-x1) (x-x1)
        (x2-x1)(y-y1) = (y2-y1)(x-x1)
        (x2-x1)(y-y1) - (y2-y1)(x-x1) = 0
        a = y1 - y2
        b = x2 - x1
        c = (x2-x1)*y1 - (y2-y1)*x1
        """
        self.p1 = kwargs['p1']
        self.p2 = kwargs['p2']
        self.a = self.p1[1] - self.p2[1]
        self.b = self.p2[0] - self.p1[0]
        self.c = (self.p2[0] - self.p1[0])*self.p1[1] - (self.p2[1]-self.p1[1])*self.p1[0]

    def _init_standard(self, **kwargs):
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.c = kwargs['c']

    def intersect(self, other):
        """
        a1 x + b1 y = c1
        a2 x + b2 y = c2
        """
        a1, b1, c1 = self.a, self.b, self.c
        a2, b2, c2 = other.a, other.b, other.c
        denom = a1*b2 - b1*a2
        if denom == 0:
            # TODO handle this better
            if a1*c2 == a2*c1:
                return 'coincident'
            return 'parallel'
        x = (c1*b2 - b1*c2)/float(denom)
        y = (a1*c2 - c1*a2)/float(denom)
        return (x, y)

    def origin_distance(self):
        # ax + by = c
        # y = c/b - a/b x
        # y = b/a x    (perpendicular through origin)
        # -b x + a y = 0
        #
        # compute intersection of these
        #
        # denom = a*a + b*b
        # x = (c*a)/denom
        # y = (c*b)/denom
        #
        # sqrt((ccaa+ccbb)/(aa+bb)^2)
        # sqrt(ccaa+ccbb)/(aa+bb)
        # abs(c)*sqrt(aa+bb)/(aa+bb)
        # abs(c) / sqrt(aa+bb)
        return np.abs(self.c) / np.sqrt(self.b**2 + self.a**2)

    def point_distance(self, x, y):
        den = np.sqrt(self.a ** 2 + self.b ** 2)
        if den == 0:
            raise ValueError
        num = np.abs(self.a*x + self.b*y + self.c)
        return num/den


def line_segment_intersection(l1, l2):
    # http://www.cs.swan.ac.uk/~cssimon/line_intersection.html
    x1, y1 = l1[0].real, l1[0].imag
    x2, y2 = l1[1].real, l1[1].imag
    x3, y3 = l2[0].real, l2[0].imag
    x4, y4 = l2[1].real, l2[1].imag
    denom = (x4-x3)*(y1-y2) - (x1-x2)*(y4-y3)
    t1_num = (y3-y4)*(x1-x3) + (x4-x3)*(y1-y3)
    t2_num = (y1-y2)*(x1-x3) + (x2-x1)*(y1-y3)

    if denom == 0:
        return False

    t1, t2 = t1_num/denom, t2_num/denom

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return True
    return False


# TODO
class LineSegment(Line):
    def intersect(self, other):
        ix = Line.intersect(self, other)
        print('i dont know how to do this yet!')
        import ipdb; ipdb.set_trace()

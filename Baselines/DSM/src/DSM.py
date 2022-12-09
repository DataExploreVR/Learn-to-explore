import math
from shapely.geometry import Point, MultiPoint
from shapely import speedups
speedups.enable()


def calAngle(center, p1, p2):
    a = center.distance(p1)
    b = center.distance(p2)
    c = p1.distance(p2)
    theta = math.acos((a**2 + b**2 - c**2)/(2*a*b))
    return theta


def sameSymbol(a, b):
    return (a >= 0 and b >= 0)or(a <= 0 and b <= 0)


class RayRegion():

    def __init__(self, center, surrounds):
        self.center = center
        self.error = False
        if len(surrounds) < 2:
            raise ValueError(
                'The number of edge points of ray region is less than 2!')
        self.surrounds = surrounds
        self.edge_points = self.surrounds[:2]
        try:
            self.cur_angle = calAngle(
                self.center, self.edge_points[0], self.edge_points[1])
            for point in self.surrounds[2:]:
                newAngles = []
                newAngles.append(
                    calAngle(self.center, self.edge_points[0], point))
                newAngles.append(
                    calAngle(self.center, self.edge_points[1], point))
                bigger = 0
                if newAngles[0] < newAngles[1]:
                    bigger = 1
                if newAngles[bigger] > self.cur_angle:
                    self.cur_angle = newAngles[bigger]
                    self.edge_points[1-bigger] = point
        except ZeroDivisionError:
            self.error = True
        except:
            raise

    def add_surrounds(self, point):
        self.surrounds.append(point)
        newAngles = []
        newAngles.append(calAngle(self.center, self.edge_points[0], point))
        newAngles.append(calAngle(self.center, self.edge_points[1], point))
        bigger = 0
        if newAngles[0] < newAngles[1]:
            bigger = 1
        if newAngles[bigger] > self.cur_angle:
            self.cur_angle = newAngles[bigger]
            self.edge_points[1-bigger] = point

    def in_region(self, point):
        x0 = point.x
        y0 = point.y
        cx = self.center.x
        cy = self.center.y
        x1 = self.center.x - self.edge_points[0].x
        x2 = self.center.x - self.edge_points[1].x
        y1 = self.center.y - self.edge_points[0].y
        y2 = self.center.y - self.edge_points[1].y
        if x1*y2 == x2*y1:
            rate_judge = (y0-cy)*x1 == y1*(x0-cx)
            dist_judge1 = (cy <= y0 and cy >= self.edge_points[0].y) or (
                cy >= y0 and cy <= self.edge_points[0].y)
            dist_judge2 = (cx <= x0 and cx >= self.edge_points[0].x) or (
                cx >= x0 and cx <= self.edge_points[0].x)
            return rate_judge and dist_judge1 and dist_judge2
        else:
            a_is_positive = sameSymbol(
                (y2*(x0 - cx) - x2*(y0 - cy)), (x1*y2 - x2*y1))
            b_is_positive = sameSymbol(
                (y1*(x0 - cx) - x1*(y0 - cy)), (x2*y1 - x1*y2))
            return (a_is_positive and b_is_positive)

    def info(self):
        print('center:', self.center)
        print('edge points:', self.edge_points[0], self.edge_points[1])
        print('angle:', self.cur_angle)


class SingleRayRegion():

    def __init__(self, neg_point, pos_point):
        self.neg_point = neg_point
        self.pos_point = pos_point

    def in_region(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        x0 = point.x
        y0 = point.y
        x1 = self.pos_point.x
        y1 = self.pos_point.y
        x2 = self.neg_point.x
        y2 = self.neg_point.y
        rate_judge = (y0-y1)*(x2-x1) == (y2-y1)*(x0-x1)
        dist_judge1 = (x2 <= x0 and x2 >= x1) or (x2 <= x1 and x2 >= x0)
        dist_judge2 = (y2 <= y0 and y2 >= y1) or (y2 <= y1 and y2 >= y0)
        if rate_judge and dist_judge1 and dist_judge2:
            return True
        return False


class DataSpaceModel():

    def __init__(self, init_pos, init_neg):
        if not isinstance(init_pos, Point):
            init_pos = Point(init_pos)
        if not isinstance(init_neg, Point):
            init_neg = Point(init_neg)
        self.pos_points = [init_pos]
        self.neg_points = [init_neg]
        self.pos_region = MultiPoint(self.pos_points).convex_hull
        self.neg_region = [SingleRayRegion(init_neg, init_pos)]
        self.neg_is_single = True

    def in_pos_region(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        return self.pos_region.contains(point) or self.pos_region.touches(point)

    def in_neg_region(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        for sub_region in self.neg_region:
            if sub_region.in_region(point):
                return True

    def get_point_region(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        if self.in_pos_region(point):
            return 1
        elif self.in_neg_region(point):
            return -1
        else:
            return 0

    def add_pos_point(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        self.pos_points.append(point)
        self.pos_region = MultiPoint(self.pos_points).convex_hull
        if self.neg_is_single:
            self.neg_is_single = False
            self.neg_region = []
            for neg_point in self.neg_points:
                RR = RayRegion(neg_point, self.pos_points.copy())
                if RR.error == False:
                    self.neg_region.append(RR)
        else:
            for sub_region in self.neg_region:
                sub_region.add_surrounds(point)

    def add_neg_point(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        if self.neg_is_single:
            self.neg_region.append(SingleRayRegion(point, self.pos_points[0]))
            self.neg_points.append(point)
        else:
            RR = RayRegion(point, self.pos_points.copy())
            if RR.error == False:
                self.neg_region.append(RR)
                self.neg_points.append(point)

    def update_three_sets(self, D_pos, D_neg, D_uncet):
        for item in D_uncet:
            mark = self.get_point_region(item)
            if mark == 1:
                D_pos.append(item)
                D_uncet.remove(item)
            elif mark == -1:
                D_neg.append(item)
                D_uncet.remove(item)
            else:
                pass
        return (D_pos, D_neg, D_uncet)

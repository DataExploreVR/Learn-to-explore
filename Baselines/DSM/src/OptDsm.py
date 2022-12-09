import numpy as np
import math

DATAMAX = 1
PRECISION = 15

def float_equal(a,b):
    return abs(a-b) < 10**(-PRECISION)

def cal_angle(center, p1, p2):
    a = (p1[0]-center[0])**2 + (p1[1]-center[1])**2
    b = (p2[0]-center[0])**2 + (p2[1]-center[1])**2
    c = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    try:
        cos = (a+b-c)/(2*math.sqrt(a*b))
    except:
        theta = 0.0
    if float_equal(cos,1.0):
        theta = 0.0
    else:
        try:
            theta = math.acos(cos)
        except:
            print('center=',center,';p1=',p1,';p2=',p2)
            print('a=',a,';b=',b,';c=',c)
            print('a+b-c=',a+b-c)
            theta = 0.0
    return theta


class PosRegion():

    def __init__(self, init_point):
        self.points = [np.asarray(init_point)]
        self.p_num = 1
        self.precision =  PRECISION

    def in_region(self, points):
        if self.p_num == 1:
            return float_equal(points[..., 0],self.points[0][0]) & float_equal(points[...,1],self.points[0][1])
        elif self.p_num == 2:
            dist0 = np.square(points - self.points[0])
            dist0 = np.sqrt(dist0[..., 0] + dist0[..., 1])
            dist1 = np.square(points - self.points[1])
            dist1 = np.sqrt(dist1[..., 0] + dist1[..., 1])
            dist_self = np.sqrt(np.sum(np.square(self.points[1] - self.points[0])))
            return float_equal(dist0+dist1,dist_self)
        else:
            cond = np.zeros(len(points)) == 1
            for index in range(self.p_num):
                next_i = (index + 1) % self.p_num
                dist0 = np.square(points - self.points[index])
                dist0 = np.sqrt(dist0[..., 0] + dist0[..., 1])
                dist1 = np.square(points - self.points[next_i])
                dist1 = np.sqrt(dist1[..., 0] + dist1[..., 1])
                dist_self = np.sqrt(np.sum(np.square(
                    self.points[index] - self.points[next_i])))
                cond = cond | float_equal(dist0+dist1,dist_self)
            count = np.zeros(len(points))
            ray_ends = np.dstack(((DATAMAX + 1)*np.ones(len(points)), points[..., 1]))[0]
            for index in range(self.p_num):
                next_i = (index + 1) % self.p_num
                o1 = (np.cross(self.points[next_i] - self.points[index],
                               points-self.points[index])) > 0
                o2 = (np.cross(self.points[next_i] - self.points[index],
                               points-self.points[next_i])) > 0
                o3 = (np.cross(ray_ends-points, self.points[index]-points)) > 0
                o4 = (np.cross(ray_ends-points, self.points[next_i]-points)) > 0
                count += ((o1 != o2) & (o3 != o4))
            count_cond = (count % 2) == 1
            return cond | count_cond

    def add_point(self, point):
        fmt_point = np.asarray([point])
        if self.in_region(fmt_point)[0]:
            return False
        else:
            self.points.append(fmt_point[0])
            self.p_num += 1
            return True
        
    def _info(self):
        print(self.points)


class SingleRayRegion():

    def __init__(self, init_neg, init_pos):
        self.neg_point = init_neg
        self.pos_point = init_pos

    def in_region(self, points):
        x0 = points[..., 0]
        y0 = points[..., 1]
        x1 = self.pos_point[0]
        y1 = self.pos_point[1]
        x2 = self.neg_point[0]
        y2 = self.neg_point[1]
        rate_judge = float_equal((y0-y1)*(x2-x1), (y2-y1)*(x0-x1))
        dist_judge1 = ((x2 <= x0) & (x2 >= x1)) | ((x2 <= x1) & (x2 >= x0))
        dist_judge2 = ((y2 <= y0) & (y2 >= y1)) | ((y2 <= y1) & (y2 >= y0))
        return rate_judge & dist_judge1 & dist_judge2

    def _info(self):
        print('pos_point:',self.pos_point)
        print('neg_point:',self.neg_point)

class RayRegion():

    def __init__(self, neg_point, pos_points):
        self.center = neg_point
        self.surrounds = pos_points
        self.edge_points = self.surrounds[:2]
        self.cur_angle = cal_angle(
            self.center, self.edge_points[0], self.edge_points[1])
        for point in self.surrounds[2:]:
            newAngles = []
            newAngles.append(
                cal_angle(self.center, self.edge_points[0], point))
            newAngles.append(
                cal_angle(self.center, self.edge_points[1], point))
            bigger = 0
            if newAngles[0] < newAngles[1]:
                bigger = 1
            if newAngles[bigger] > self.cur_angle:
                self.cur_angle = newAngles[bigger]
                self.edge_points[1-bigger] = point

    def add_surrounds(self, pos_point):
        self.surrounds.append(pos_point)
        newAngles = []
        newAngles.append(
            cal_angle(self.center, self.edge_points[0], pos_point))
        newAngles.append(
            cal_angle(self.center, self.edge_points[1], pos_point))
        bigger = 0
        if newAngles[0] < newAngles[1]:
            bigger = 1
        if newAngles[bigger] > self.cur_angle:
            self.cur_angle = newAngles[bigger]
            self.edge_points[1-bigger] = pos_point
            return True
        return False

    def in_region(self, points):
        a = np.asarray([self.center[0]-self.edge_points[0][0],
                        self.center[1]-self.edge_points[0][1]])
        b = np.asarray([self.center[0]-self.edge_points[1][0],
                        self.center[1]-self.edge_points[1][1]])
        if np.cross(a,b) <= 10**(-PRECISION):
            x0 = points[..., 0]
            y0 = points[..., 1]
            x1 = self.edge_points[0][0]
            y1 = self.edge_points[0][1]
            x2 = self.center[0]
            y2 = self.center[1]
            rate_judge = float_equal((y0-y1)*(x2-x1),(y2-y1)*(x0-x1))
            dist_judge1 = ((x2 <= x0) & (x2 >= x1)) | ((x2 <= x1) & (x2 >= x0))
            dist_judge2 = ((y2 <= y0) & (y2 >= y1)) | ((y2 <= y1) & (y2 >= y0))
            return rate_judge & dist_judge1 & dist_judge2
        else:
            center = np.dstack((points[..., 0]-self.center[0],
                                points[..., 1]-self.center[1]))[0]
            symbol1 = np.cross(a, center) >= 0
            symbol2 = np.cross(center, b) >= 0
            symbol0 = np.cross(a, b) >= 0
            return (symbol1 == symbol0) & (symbol2 == symbol0)

    def _info(self):
        print('center:', self.center)
        print('edge points:', self.edge_points[0], self.edge_points[1])
        print('angle:', self.cur_angle)


class OptDataSpaceModel():

    def __init__(self, init_pos, init_neg):
        self.pos_points = [init_pos]
        self.neg_points = [init_neg]
        self.pos_region = PosRegion(init_pos)
        self.neg_region = [SingleRayRegion(init_neg, init_pos)]
        self.neg_is_single = True
        self.pos_modified = True
        self.neg_modified = [True]

    def in_pos_region(self, points):
        return self.pos_region.in_region(points)

    def in_neg_region(self, points):
        in_region = np.ones(len(points)) == 0
        for sub_region in self.neg_region:
            in_region = in_region | sub_region.in_region(points)
        return in_region

    def add_pos_point(self, point):
        self.pos_points.append(point)
        self.pos_modified = self.pos_region.add_point(point)
        if self.neg_is_single:
            self.neg_is_single = False
            self.neg_modified = [True]*len(self.neg_points)
            self.neg_region = []
            for neg_point in self.neg_points:
                self.neg_region.append(RayRegion(neg_point, self.pos_points))
        else:
            for i in range(len(self.neg_points)):
                self.neg_modified[i] = self.neg_region[i].add_surrounds(point)

    def add_neg_point(self, point):
        self.pos_modified = False
        self.neg_modified = [False]*len(self.neg_points)
        self.neg_modified.append(True)
        if self.neg_is_single:
            self.neg_region.append(SingleRayRegion(point, self.pos_points[0]))
            self.neg_points.append(point)
        else:
            self.neg_region.append(RayRegion(point, self.pos_points))
            self.neg_points.append(point)

    def get_single_point_region(self, point):
        if self.in_pos_region(np.asarray([point]))[0]:
            return 1
        elif self.in_neg_region(np.asarray([point]))[0]:
            return -1
        else:
            return 0

    def get_points_region(self,points):
        in_pos = self.in_pos_region(points)
        in_neg = self.in_neg_region(points)
        if np.sum((in_pos ==True) & (in_neg==True))>=1:
            print(points[(in_pos ==True) & (in_neg==True)])
            self._info()
            raise ValueError('Point in both regions! Check the precision!')
        regions = (in_pos==True) + 0
        regions = regions - (in_neg==True)
        return regions


    def update_three_sets(self, D_pos, D_neg, D_uncet):
        if not isinstance(D_uncet, np.ndarray):
            D_uncet = np.asarray(D_uncet)

        if self.pos_modified:
            cond = self.pos_region.in_region(D_uncet)
            D_pos.extend(D_uncet[cond])
            D_uncet = D_uncet[~cond]
        for i in range(len(self.neg_modified)):
            if self.neg_modified[i]:
                cond = self.neg_region[i].in_region(D_uncet)
                D_neg.extend(D_uncet[cond])
                D_uncet = D_uncet[~cond]
        return D_pos,D_neg,D_uncet

    def _info(self):
        print('pos region:')
        self.pos_region._info()
        print('neg regions:')
        for neg_region in self.neg_region:
            if isinstance(neg_region,RayRegion):
                print('RayRegion:')
            else:
                print('Single RayRegion:')
            neg_region._info()
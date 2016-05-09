import kivy
kivy.require('1.9.1') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
#from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.core.image import Image
#from kivy.graphics import BorderImage
from kivy.graphics import Color, Rectangle
#from kivy.uix.image import AsyncImage

import numpy as np
import math
import collections
'''
A parking lot and a car finding empty slots
+--+--+--+
|  |  |  |
|  |  |  |
+--+--+--+

+--+
+--+

if the sensor reading comes from one of the parking lot,
then mark the whole parking lot as occupied,
else mark the grid as occupied
'''

PARKINGTHRESHOLD = 0.5 # parking threshold

# [(a,b), r, (tstart, tend)] all in local coordinate in meters
# counter clockwise
RADAR = {
    "leftfront": [(0.5, 0.5), 5.2, (20, 110)],
    "leftrear": [(-3.8, 0.5), 5.2, (70, 160)],
    "rearleft": [(-3.8, 0.5), 5.2, (135, 215)],

    "rightfront": [(0.5, -0.5), 5.2, (250, 340)],
    "rightrear": [(-3.8, -0.5), 5.2, (200, 290)],
    "rearright": [(-3.8, -0.5), 5.2, (135, 225)]
}

def lineCross(l1, l2):
    # line segments
    [(a,b), (c,d)], [(e,f), (g,h)] = l1, l2
    # parrallel case
    if (h-f)*(c-a) == (d-b)*(g-e): return False
    # find intersection of the line, solve for the parametric equation for t1 and t2
    # ( g-e, a-c ) (t1)  = ( a-e )
    # ( h-f, b-d ) (t2)    ( b-f )
    A = np.array([[g-e, a-c],
                  [h-f, b-d]])
    y = np.array([a-e, b-f])
    t1, t2 = np.linalg.lstsq(A, y)[0]
    # test if intersection is in both line
    return (0 <= t1 <= 1) and (0 <= t2 <= 1)

# def inFan(carState, position, x, y):
#     '''test to see if x, y is in a fan'''

#     def localToGlobal(coord):
#         '''
#         from car coordinate to global coordinate
#         '''
#         x, y = coord
#         theta = -carState.theta # - for counterclockwise
#         # rotate (x, y) by carState.theta around carState.x, carState.y
#         dx = x - carState.x
#         dy = y - carState.y
#         rotMatrix = [[math.cos(theta), -math.sin(theta)],
#                      [math.sin(theta), math.cos(theta)]]
#         return (math.cos(theta)*dx - math.sin(theta)*dy + x, math.sin(theta)*dx + math.cos(theta)*dy + y)

#     params = RADAR.get(position, None)
#     if not params: exit(1)
#     o, r, (tstart, tend) = params
#     ox, oy = localToGlobal(o)
#     tstart = TODO

#     # not in range
#     if ((x-ox)**2 + (y-oy)**2) ** 0.5 > r: return False
#     # not in theta
#     theta = (math.atan2(y-oy, x-ox) + math.pi) / 2*math.pi * 360
#     if theta < tstart or theta > tend: return False

#     return True


def inConvexPolygon(vertices, pt):
    '''
    test if pt is in the convex polygon defined by edges, where vertices are aranged counterclockwise
    vertices: [(x1, y1), (x2, y2), (x3, y3) ... ]
    edge: [(x1, y1), (x2, y2)] from x1 to x2
    '''
    def onLeft(edge):
        xp, yp = pt
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        # perpendicular to the edge
        A = - (y2 - y1)
        B = x2 - x1
        # (A,B) * (x1, y1) + C = 0 =>
        C = -A*x1 - B*y1
        return (A * xp + B * yp + C >= 0)

    for i in range(len(vertices)):
        edge = [vertices[i], vertices[(i+1) % len(vertices)]]
        if not onLeft(edge): return False

    return True

class LogOddSquareGrid:
    '''
    +-------------+ (side_, side_)
    |      |side_ |
    |      |      |
    |    (0, 0)   |
    |             |
    |             |
    +-------------+ (side_, -side_)
    assumes the grid is square and the content of grid is logOdds
    assumes grid coordinate is scaled and translated version of global coordinate

    @params
    origin_global: global origin
    side_: half the side_length of the grid
    step_: side_length for each cell in meter
    translate_: translate 2 tuple
    l0_: initial belief
    parking_: list of parking lots [[(1,2),(3,4),(5,6),(7,8)], ... ]
    parking_status_: logodds of parking lot
    '''
    def __init__(self, origin_global=(0,0), side=2, step=0.5, l0=0):
        self.side_ = side
        self.step_ = step
        self.translate_ = origin_global
        self.cell_ = np.zeros([2*side+1, 2*side+1], dtype='double')
        self.l0_ = l0
        self.parking_ = []
        self.parking_status_ = np.zeros(10)
        self.initParkingLot("pilot_grid_map.csv")

    def reset(self):
        self.cell_.fill(self.l0_)
        self.parking_ = []
        self.parking_status_ = np.zeros(10)

    def initParkingLot(self, filename):
        # readin pilot_grid_map.csv, in global position
        with open(filename) as f:
            header = True;
            for l in f:
                if header: header = False; continue
                se1, se2, sw1, sw2, ne1, ne2, nw1, nw2 = map(lambda x: float(x), l.split(',')[1:])
                self.parking_.append([(sw1,sw2), (se1,se2), (ne1,ne2), (nw1, nw2)])
        self.parking_status_ = np.zeros(len(self.parking_))

    def cellToGlobal(self, coord):
        '''from cell coordinate center to utm17 global coordinate'''
        x, y = coord
        tx, ty = self.translate_
        return (self.step_ * x + tx + self.step_ / 2, self.step_ * y + ty + self.step_ / 2)

    def globalToCell(self, coord):
        '''from utm17 global coordinate'''
        x, y = coord
        tx, ty = self.translate_
        return ( int ((x - tx) / self.step_), int((y - ty) / self.step_))


class State:
    '''position and oriention of the car'''
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta

class Mapping:
    def __init__(self, l0=0, locc=10, lfree=-10):
        self.l0_ = l0
        self.locc_ = locc
        self.lfree_ = lfree
        self.grid_ = LogOddSquareGrid(l0=l0)
        self.X_ = State() # state given by localizaiton in global coordinate
        self.updated_parking_ = np.zeros(len(self.grid_.parking_))

    def updateGrid(self, localx, localy, position, refresh_update=True):
        '''
        update log odds on grid after sensor reading comes in
        update rule: l_t = log (P(m_i | z_t) / P(-m_i | z_t)) - l_0 + l_{t-1}
        refresh_update: whether set updated_parking_ to 0
        '''
        if refresh_update: self.updated_parking_.fill(0)

        # get where the car is
        cellX, cellY = self.grid_.globalToCell((self.X_.x, self.X_.y))
        x, y = cellX + localx, cellY + localy

        # updated_parking = np.zeros(self.grid_.parking_status_)

        # mark parking lot corresponding to localx and localy
        for i, vertices in enumerate(self.grid_.parking_):
            if inConvexPolygon(vertices, (x, y)):
                self.grid_.parking_status_[i] = self.grid_.parking_status_[i] + self.locc_ - self.l0_
                self.updated_parking_[i] = 1
                break

        # mark the cell containing the datapoint as occupied, in fact, we don't care as we only care about whether parking lot is empty
        # self.grid_.cell_[x][y] = self.grid_.cell_[x][y] + self.locc_ - self.l0_

        # mark intersecting parking lot with sensor white if enough area covered
        candidateDict = collections.defaultdict(list) # {parking_lot_num: [sensor_position1, sensor_position2]}

        v = RADAR[position]
        fanVertices = self.getFanVertices(v)
        for i, vertices in enumerate(self.grid_.parking_):
            # if it updated in the interval of time then it's not a candidate
            if self.updated_parking_[i]: continue
            # test if any line form fan crosses with the given parking lot
            isCand = False
            for i in range(len(fanVertices)):
                for j in range(len(vertices)):
                    if lineCross([fanVertices[i], fanVertices[(i+1)%len(fanVertices)]],
                                 [vertices[j], vertices[(j+1)%len(vertices)]]):
                        candidateDict[i].append(position)
                        isCand = True
                        break
                    if isCand: break

        # for each cand in candidateDict
        # find the portion covered by both the sensor and the parking lot
        for parking_num, sensorList in candidateDict:
            if self.getAreaRatio(parking_num, sensorList) >= PARKINGTHRESHOLD:
                # mark free
                self.grid_.parking_status_[parking_num] = self.grid_.parking_status_[i] + self.free_ - self.l0_

    def getFanVertices(self,v): # v is value in RADAR
        (cx, cy), r, (tstart, tend) = v

        # first convert to global location
        gx = self.X_.x + cx * math.cos(self.X_.theta) + cy * math.sin(self.X_.theta)
        gy = self.X_.y - cx * math.sin(self.X_.theta) + cy * math.cos(self.X_.theta)
        cx, cy = gx, gy
        tstart = tstart + self.X_.theta
        tend = tend + self.X_.theta
        # convert to cell position
        (cx, cy) = self.grid_.globalToCell((cx, cy))

        tmid = (tstart + tend) / 2.0
        # filter out unwanted parking lot using approximation: 16 comparison at most for each parking lot

        return [(cx, cy), (cx+r*math.cos(tstart), cy+r*math.sin(tstart)),
                (cx+r*math.cos(tmid), cy+r*math.sin(tmid)), (cx+r*math.cos(tend), cy+r*math.sin(tend))]


    def getAreaRatio(self, parking_num, sensorList):
        '''return the percentage of parking area covered by sensors'''
        # get the parking lot vertices
        parkingLot = self.grid_.parking_[parking_num]
        # get its surrounding rectangle box
        xList = map(lambda p: p[0], parkingLot)
        yList = map(lambda p: p[1], parkingLot)

        # convert to cell coordinate
        minX, minY = self.grid_.globalToCell((min(xList), min(ylist)))
        maxY, maxY = self.grid_.globalToCell((max(xList), max(ylist)))

        # calculate area
        denominator = 0.0 # area of parking lot
        numerator = 0.0 # area of intersection(parking lot, sensors)
        for x in xrange(minX, maxX+1):
            for y in xrange(minY, maxY+1):
                # pt in parking lot
                if inConvexPolygon(parkingLot, (x,y)):
                    denominator = denominator + 1
                    for pos in sensorList:
                        fan = getFanVertices(RADAR[pos])
                        if inConvexPolygon(fan, (x, y)):
                            numerator = numerator + 1
                            break

        return numerator / denominator


    def handleSensor(self): # handler for reading data to call
        '''read in local sensor reading'''
        # for each timestamp reading, updateGrid once, skip header
        f_trace = open("UTM_trace.csv");
        f_trace.readline()

        sensor_names = ["leftfront", "leftrear", "rearleft", "rightfront", "rightrear", "rearright"]
        sensors = map(lambda x: open(x + ".csv"), sensor_names)
        unprocessed_reading = map(lambda x: None, sensor_names) # [None, (t,lx,ly), None, ...]

        for f in sensors: f.readline() # skip headers

        gx,gy,_,_,t0,speed,_,_,theta,_,_,_ = map(lambda x: float(x), f_trace.readline().split(','))
        t1 = t0
        for l in f_trace:

            print l
            # localization goes here: todo
            self.updateState(State(gx,gy,theta))
            t0 = t1
            new_interval = True
            # new current time
            gx,gy,_,_,t1,speed,_,_,theta,_,_,_ = map(lambda x: float(x), l.split(','))

            # read in sensor reading up to this timestamp
            for i, f in enumerate(sensors):

                # deal with unprocessed sensor reading
                if unprocessed_reading[i]:
                    t, lx, ly = unprocessed_reading[i]
                    if t<=t1:
                        self.updateGrid(lx,ly,sensor_names[i], new_interval)
                        new_interval = False
                        unprocessed_reading[i] = None
                    else: # not yet resolved, so don't read in new readings
                        continue

                #  read in new sensor reading
                while True:
                    line = f.readline()
                    if not line: break # no more reading
                    _,_,t,_,_,lx,ly,_,_ = map(lambda x: float(x), line.split(','))
                    # make sure timestamp is between t0 and t1: todo
                    if t <= t1:
                        self.updateGrid(lx,ly,sensor_names[i], new_interval)
                        new_interval = False
                    else:
                        unprocessed_reading[i] = (t,lx,ly)
                        break # wait for new iteration

        f_trace.close()
        for f in sensors: f.close()

    def updateState(self, X): # handler for localization to call
        self.X_ = X


class StartScreen(Screen):
    pass

class GameScreen(Screen):
    pass

class RootScreen(ScreenManager):
    pass


class MainApp(App):
    def build(self):
        return RootScreen()


if __name__ == '__main__':
    MainApp().run()
    m = Mapping() # plz give the global origin coordinate
    m.handleSensor()


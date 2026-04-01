from scipy import stats, interpolate
import numpy as np
import math as m
class GeometryMethod():
    def __init__(self, theta_floor):
        self.theta_floor = theta_floor * (m.pi/180)

    def detect(self,track):
        lenA = m.floor(len(track)/2)
        #print("track: ",track)
        self.trackA = track[0:lenA]
        self.trackB = track[lenA:]
        #print("trackA: ",self.trackA)
        #print("trackB: ",self.trackB)
        try:
            self.lineA = self.get_line(self.trackA)
            self.lineB = self.get_line(self.trackB)
            self.theta = self.get_angle(self.lineA, self.lineB)
        except:
            self.theta = 0
            print("Warning: unable to get angle")

        if self.theta > self.theta_floor:
            return True
        else:
            return False

    def get_line(self, points):
        x, _ = zip(*points)
        #print("x: ",x)
        _, y = zip(*points)
        #print("y: ",y)
        regression = stats.linregress(x,y)
        return regression

    def get_angle(self, lineA, lineB):
        theta = m.atan2((lineA.slope - lineB.slope),(1 + (lineA.slope * lineB.slope)))
        return theta
    
class DerivativeMethod():
    def __init__(self, floor):
        self.floor = floor
        
    def split(self, points):
        x, _ = zip(*points)
        print("x: ",x)
        _, y = zip(*points)
        print("y: ",y)
        return np.asarray(x), np.asarray(y)
        
    def detect(self,track):
        x, y = self.split(track)
        print(track)
        try: 
            spline = interpolate.make_splrep(x,y,s=0)
            der = spline(x,2)
            if abs(der.any()) > self.floor:
                return True
        except:
            print("Warning: unable to get derivative")
            return False
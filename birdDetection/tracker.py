from scipy import stats
import math as m
class BirdTrack():
    def __init__(self, initial_point):
        self.track = []
        self.track.append(initial_point)

    def add_point(self, point):
        self.track.append(point)

    def geometry_track(self, n_points, theta_floor):
        COLLISION = GeometryMethod(self.track,n_points,theta_floor)
        return COLLISION 
    
class GeometryMethod():
    def __init__(self, track, n_points, theta_floor):
        theta_floor = theta_floor * (m.pi/180)
        lenTrack = len(track) - 1
        lenA = m.floor(n_points/2)
        self.trackA = []
        self.trackB = []
        for i in track[lenTrack: lenTrack - lenA : -1]:
            self.trackA.append(i)
        for i in track[lenTrack - lenA - 1: lenTrack - n_points : -1]:
            self.trackB.append(i)
        self.lineA = self.get_line(self.trackA)
        self.lineB = self.get_line(self.trackB)
        self.theta = self.get_angle(self.lineA, self.lineB)
        if self.theta < theta_floor:
            return True
        else:
            return False

    def get_line(self, points):
        x = points[:0]
        y = points[:1]
        regression = stats.linregress(x,y)
        return regression

    def get_angle(self, lineA, lineB):
        theta = m.atan2((lineA.slope - lineB.slope),(1 + (lineA.slope * lineB.slope)))
        return theta
    

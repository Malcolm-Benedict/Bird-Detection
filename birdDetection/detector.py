from scipy import stats
import math as m
class GeometryMethod():
    def __init__(self, theta_floor):
        """Detector object

        Args:
            theta_floor (float): Floor to detect in degrees.
        """
        self.theta_floor = theta_floor * (m.pi/180)

    def detect(self,track):
        """Split track into two sections and determine the angle between the sections.

        Args:
            track (list[]): Ordered list of points

        Returns:
            bool: detection flag
        """
        # Split track
        lenA = m.floor(len(track)/2)
        self.trackA = track[0:lenA]
        self.trackB = track[lenA:]
        
        # Ensure tracks are long enough
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
        """Make a line from points

        Args:
            points (list[]): points

        Returns:
            LinregressResult: line object
        """
        x, _ = zip(*points)
        _, y = zip(*points)
        regression = stats.linregress(x,y)
        return regression

    def get_angle(self, lineA, lineB):
        """Gets the angle between the lines

        Args:
            lineA (LinregressResult): First half
            lineB (LinregressResult): Second Half

        Returns:
            float: theta
        """
        theta = m.atan2((lineA.slope - lineB.slope),(1 + (lineA.slope * lineB.slope)))
        return theta
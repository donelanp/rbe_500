import math
import numpy as np
from enum import Enum

class JointType(Enum):
    PRISMATIC = 1
    REVOLUTE  = 2

class Link():
    def __init__(self, linkOffset, linkLength, linkTwist, jointAngle, jointType):
        self.linkOffset = linkOffset
        self.linkLength = linkLength
        self.linkTwist  = linkTwist
        self.jointAngle = jointAngle
        self.jointType  = jointType

    def HomogeneousMatrix(self):
        # temporary helper variables
        ctheta = math.cos(self.jointAngle)
        stheta = math.sin(self.jointAngle)
        calpha = math.cos(self.linkTwist)
        salpha = math.sin(self.linkTwist)

        # round to zero
        if abs(ctheta) < 1e-6: ctheta = 0
        if abs(stheta) < 1e-6: stheta = 0
        if abs(calpha) < 1e-6: calpha = 0
        if abs(salpha) < 1e-6: salpha = 0

        # populate homogenous transformation matrix
        A = np.zeros((4, 4))
        A[0,0] = ctheta
        A[1,0] = stheta

        A[0,1] = -calpha * stheta
        A[1,1] =  calpha * ctheta
        A[2,1] =  salpha

        A[0,2] =  salpha * stheta
        A[1,2] = -salpha * ctheta
        A[2,2] =  calpha

        A[0,3] = self.linkLength * ctheta
        A[1,3] = self.linkLength * stheta
        A[2,3] = self.linkOffset
        A[3,3] = 1

        return A

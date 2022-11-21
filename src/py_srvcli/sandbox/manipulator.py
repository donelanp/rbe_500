import numpy as np
from link import JointType

class Manipulator:
    def __init__(self, links):
        self.links = links

    def ForwardKinematics(self):
        return self.SubForwardKinematics(0, len(self.links))

    def VelocityKinematics(self, jointVelocities):
        return self.Jacobian() @ jointVelocities

    def Jacobian(self):
        J = np.zeros((6, len(self.links)))

        # end effector pose represented in inertial frame
        Tn = self.ForwardKinematics()

        for ii in range(0, len(self.links)):
            # link ii-1 pose represented in inertial frame
            if ii == 0: Tprev = np.identity(4)
            else: Tprev = self.SubForwardKinematics(1, ii-1)

            # link ii's contribution to the manipulator jacobian
            if self.links[ii].jointType == JointType.PRISMATIC:
                J[0:3,ii] = Tprev[0:3,2]
            elif self.links[ii].jointType == JointType.REVOLUTE:
                J[0:3,ii] = np.cross(Tprev[0:3,2], Tn[0:3,3] - Tprev[0:3,3])
                J[3:6,ii] = Tprev[0:3,2]
            else: raise Exception('Unhandled joint type')

        return J

    def SubForwardKinematics(self, startIndex, endIndex):
        T = self.links[startIndex].HomogeneousMatrix()
        for ii in range(startIndex + 1, endIndex):
            T = T @ self.links[ii].HomogeneousMatrix()
        return T

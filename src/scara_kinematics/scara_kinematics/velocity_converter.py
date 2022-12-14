# call service using: ros2 service call /JointVelToEndEffectorVel interface_pkg/srv/VelocityConversion "{joint_states:[0,0,0],velocity:[1,2,3]}"
# call service using: ros2 service call /EndEffectorVelToJointVel interface_pkg/srv/VelocityConversion "{joint_states:[0,0,0],velocity:[0,3.9,-3,0,0,3]}"

import array
import math
import numpy as np
import rclpy
import sympy as sym
import warnings
from enum import Enum
from interface_pkg.srv import VelocityConversion
from rclpy.node import Node

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

class Manipulator:
    def __init__(self, links):
        self.links = links

    def ForwardKinematics(self):
        return self.SubForwardKinematics(0, len(self.links))

    def SubForwardKinematics(self, startIndex, endIndex):
        T = self.links[startIndex].HomogeneousMatrix()
        for ii in range(startIndex + 1, endIndex):
            T = T @ self.links[ii].HomogeneousMatrix()
        return T

    def Jacobian(self):
        J = np.zeros((6, len(self.links)))

        # end effector pose represented in inertial frame
        Tn = self.ForwardKinematics()

        for ii in range(0, len(self.links)):
            # previous link's pose represented in inertial frame
            if ii == 0: Tprev = np.identity(4)
            else: Tprev = self.SubForwardKinematics(0, ii)

            # current link's contribution to the manipulator jacobian
            if self.links[ii].jointType == JointType.PRISMATIC:
                J[0:3,ii] = Tprev[0:3,2]
            elif self.links[ii].jointType == JointType.REVOLUTE:
                J[0:3,ii] = np.cross(Tprev[0:3,2], Tn[0:3,3] - Tprev[0:3,3])
                J[3:6,ii] = Tprev[0:3,2]
            else: raise Exception('Unhandled joint type')

        return J

class VelocityConverter(Node):
    def __init__(self, width, length0, length1, length2):
        super().__init__('velocity_converter')
        self.jointToEndEffectorSrv = self.create_service(VelocityConversion, 'JointVelToEndEffectorVel', self.JointToEndEffector)
        self.endEffectorToJointSrv = self.create_service(VelocityConversion, 'EndEffectorVelToJointVel', self.EndEffectorToJoint)

        self.width_   = width
        self.length0_ = length0
        self.length1_ = length1
        self.length2_ = length2
    
    def Jacobian(self, jointValues):
        # joint variables
        t1 = jointValues[0]
        t2 = jointValues[1]
        d3 = jointValues[2]

        # manipulator links
        l1 = Link(         self.width_, self.length1_ - self.width_,       0, t1, JointType.REVOLUTE)
        l2 = Link(                   0,               self.length2_, math.pi, t2, JointType.REVOLUTE)
        l3 = Link(d3 + self.width_ / 2,                           0,       0,  0, JointType.PRISMATIC)

        # manipulator
        m  = Manipulator([l1, l2, l3])

        return m.Jacobian()

    def JointToEndEffector(self, request, response):
        # calculate jacobian given current joint states
        J = self.Jacobian(request.joint_states)

        # calculate end effector velocities [linear; angular] using jacobian and joint velocities
        v = J @ np.array([[request.velocity[0]],[request.velocity[1]],[request.velocity[2]]])

        # add end effector velocity to response
        response.velocity = array.array('f', [v[0], v[1], v[2], v[3], v[4], v[5]])

        return response

    def EndEffectorToJoint(self, request, response):
        # calculate jacobian given current joint states
        J = self.Jacobian(request.joint_states)

        # create augmented matrix
        v = np.array([[request.velocity[0]],[request.velocity[1]],[request.velocity[2]],\
            [request.velocity[3]],[request.velocity[4]],[request.velocity[5]]])
        Jaug = np.concatenate((J, v), axis=1)

        # put augmented matrix in rref in order to get joint velocities
        if np.linalg.matrix_rank(J) == np.linalg.matrix_rank(Jaug):
            rref = sym.Matrix(Jaug).rref()

            # add joint velocity to response
            response.velocity = array.array('f', [rref[0][0,3], rref[0][1,3], rref[0][2,3]])
        else:
            warnings.warn('rank(J) != rank(J|v)')

        return response

def main(args=None):
    rclpy.init(args=args)
    velocity_pd_controller = VelocityConverter(0.1, 1, 1, 1)
    rclpy.spin(velocity_pd_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
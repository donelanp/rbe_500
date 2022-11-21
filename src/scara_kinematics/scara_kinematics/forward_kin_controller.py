import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState 
from geometry_msgs.msg import Pose

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

class SCARA(Manipulator):
    def __init__(self, links):
        assert len(links) == 3, '3 links required'
        assert links[0].jointType == JointType.REVOLUTE, 'joint 1 must be revolute'
        assert links[1].jointType == JointType.REVOLUTE, 'joint 2 must be revolute'
        assert links[2].jointType == JointType.PRISMATIC, 'joint 3 must be revolute'
        self.links = links

class ForwardControllerNode(Node):
    def __init__(self, width, length0, length1, length2):
        super().__init__('forward_kin_controller')
        self.joint_subscriber_ = self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.cmd_end_effector_publisher_ = self.create_publisher(Pose, "/end_effector_values", 10)    


        self.width_   = width
        self.length0_ = length0
        self.length1_ = length1
        self.length2_ = length2

    def rot_mat_to_quat(self, local_T):
        sum = np.matrix.trace(local_T) # return the sum along diagonals of the array.
        buf = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64) #create buffer array

        if(sum > 0):
            sum = np.sqrt(sum + 1)
            buf[3] = 0.5 * sum
            sum = 0.5/sum
            buf[0] = (local_T[2,1] - local_T[1,2]) * sum
            buf[1] = (local_T[0,2] - local_T[2,0]) * sum
            buf[2] = (local_T[1,0] - local_T[0,1]) * sum

        else:
            i = 0
            if (local_T[1,1] > local_T[0,0]):
                i = 1
            if (local_T[2,2] > local_T[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            sum = np.sqrt(local_T[i,i] - local_T[j,j] - local_T[k,k] + 1)
            buf[i] = 0.5 * sum
            sum = 0.5 /sum
            buf[3] = (local_T[k,j] - local_T[j,k]) * sum
            buf[j] = (local_T[j,i] + local_T[i,j]) * sum
            buf[k] = (local_T[k,i] + local_T[i,k]) * sum

        return buf

    def joint_callback(self, msg):
        assert len(msg.position) == 3, 'message should contain 3 joint values'
        T = self.ForwardKinematics(*msg.position)
        #self.get_logger().info(f'\n{T}')
        temp_rot = T[0:3,0:3]
        rot_mat_to_quat_buffer = self.rot_mat_to_quat(temp_rot)

        flatlist=[element for sublist in T for element in sublist]  #converting T to one dimension array

        final_T = flatlist[3], flatlist[7], flatlist[11]        #returning end effector values
        mat = Pose()

        mat.position.x = final_T[0]
        mat.position.y = final_T[1]
        mat.position.z = final_T[2]
        mat.orientation.x = rot_mat_to_quat_buffer[0]
        mat.orientation.y = rot_mat_to_quat_buffer[1]
        mat.orientation.z = rot_mat_to_quat_buffer[2]
        mat.orientation.w = rot_mat_to_quat_buffer[3]

        self.cmd_end_effector_publisher_.publish(mat)
    
    def ForwardKinematics(self, *args):
        # joint variables
        t1 = args[0]
        t2 = args[1]
        d3 = args[2]

        # manipulator links
        l1 = Link(         self.width_, self.length1_ - self.width_,       0, t1, JointType.REVOLUTE)
        l2 = Link(                   0,               self.length2_, math.pi, t2, JointType.REVOLUTE)
        l3 = Link(d3 + self.width_ / 2,                           0,       0,  0, JointType.PRISMATIC)

        # manipulator
        m  = SCARA([l1, l2, l3])    
        T = m.ForwardKinematics()
        
        # account for base link
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.length0_ + self.width_ / 2], [0, 0, 0, 1]]) @ T
            
        return T

def main(args=None):
    rclpy.init(args=args)
    forward_kin_controller = ForwardControllerNode(0.1, 1, 1, 1)
    rclpy.spin(forward_kin_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
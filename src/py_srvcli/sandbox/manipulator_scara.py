import math
import numpy as np
from manipulator import Manipulator

class SCARA(Manipulator):
    def __init__(self, links):
        assert len(links) == 3, 'too many links'
        self.links = links

    def InverseKinematics(self, position):
        assert len(self.links) == 3, 'too many links'
        assert len(position) == 3, 'desired position must have 3 elements'

        # desired values
        x = position[0]
        y = position[1]
        z = position[2]

        # fixed parameters
        d1 = self.links[0].linkOffset
        a1 = self.links[0].linkLength
        a2 = self.links[1].linkLength

        # helper variables
        D = (x*x + y*y - a1*a1 - a2*a2) / (2 * a1 * a2)

        # joint variables
        d3 = np.array([d1 - z, d1 - z])
        t2 = np.arctan2(np.array([math.sqrt(1 - D*D), -math.sqrt(1 - D*D)]), D)
        t1 = -np.arctan2(a2 * np.sin(t2), a1 + a2 * np.cos(t2)) + np.arctan2(y, x)

        # round joint variables to zero
        d3[abs(d3) < 1e-6] = 0
        t2[abs(t2) < 1e-6] = 0
        t1[abs(t1) < 1e-6] = 0

        return np.array([t1, t2, d3]).T

if __name__ == '__main__':
    import math
    from link import JointType, Link

    # fixed parameters
    width   = 0.1
    length0 = 1
    length1 = 1
    length2 = 1

    # joint variables
    # t1 = 2.16
    # t2 = 0.46
    # d3 = 0.59
    t1 = 1.57 
    t2 = 1.57
    d3 = 0

    # manipulator links
    l1 = Link(         width, length1 - width,       0, t1, JointType.REVOLUTE)
    l2 = Link(             0,         length2, math.pi, t2, JointType.REVOLUTE)
    l3 = Link(d3 + width / 2,               0,       0,  0, JointType.PRISMATIC)

    # manipulator
    m  = SCARA([l1, l2, l3])

    T = m.ForwardKinematics()
    q = m.InverseKinematics(T[0:3,3])
    
    # account for base link
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, length0 + width / 2], [0, 0, 0, 1]]) @ T

    # account for tool link
    q[:,2] -= width / 2

    print(f'q = {[t1, t2, d3]}')
    print(f'T =\n{T}')
    print(f'q_hat =\n{q}')

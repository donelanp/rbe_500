from interface_pkg.srv import InvKin

import rclpy
from rclpy.node import Node
import numpy as np
import math

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(InvKin, 'InvKin', self.InvKin_callback)

    def InvKin_callback(self, request, response):
        # desired values
        x = request.xyz[0]
        y = request.xyz[1]
        z = request.xyz[2]
        self.get_logger().info('Incoming request\nxyz: [%.2f,%.2f,%.2f]' % (x, y, z))

        # fixed parameters
        d1 = 1.0 + 0.1
        a1 = 1.0 - 0.1
        a2 = 1.0

        # helper variables
        D = (x*x + y*y - a1*a1 - a2*a2) / (2 * a1 * a2)

        # joint variables
        q3 = np.array([d1-z, d1-z])
        q2 = np.arctan2(np.array([math.sqrt(1 - D*D), -math.sqrt(1 - D*D)]), D)
        q1 = -np.arctan2(a2 * np.sin(q2), a1 + a2 * np.cos(q2)) + np.arctan2(y, x)

        # round joint variables to zero
        q3[np.abs(q3) < 1e-6] = 0
        q2[np.abs(q2) < 1e-6] = 0
        q1[np.abs(q1) < 1e-6] = 0
        
        response.q1 = np.ndarray.tolist(q1)
        response.q2 = np.ndarray.tolist(q2)
        response.q3 = np.ndarray.tolist(q3)

        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

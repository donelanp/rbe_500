import sys

from interface_pkg.srv import InvKin
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(InvKin, 'InvKin')
        while not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = InvKin.Request()

    def send_request(self, x, y, z):
        self.req.xyz = [x,y,z]
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    num_solutions = len(response.q1)
    for i in range(num_solutions):
        minimal_client.get_logger().info(
            'Result %d: q1 = %.2f, q2 = %.2f, q3 = %.2f' %
            (i, response.q1[i], response.q2[i], response.q3[i]))

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

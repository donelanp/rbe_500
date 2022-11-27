import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class PDControllerNode(Node):
    def __init__(self, update_rate, kp, kd):
        # ros node setup (service, subscriber, publisher, timer)
        super().__init__('pd_controller')
        self.joint_subscriber_ = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.effort_publisher_ = self.create_publisher(Float64MultiArray, '/forward_effort_controller/commands', 10)
        self.timer_ = self.create_timer(update_rate, self.publish_effort)

        # update rate for effort
        self.update_rate_ = update_rate

        # proportional gain (3x1 vector)
        self.kp_ = kp

        # derivative gain (3x1 vector)
        self.kd_ = kd

        # reference (goal) state of joints (3x1 vector)
        self.ref_state_ = np.zeros((3,1))

        # current state of joints (3x1 vector)
        self.cur_state_ = np.zeros((3,1))

        # previous state error of joints (3x1 vector)
        self.prev_error_ = np.zeros((3,1))

    def joint_callback(self, msg):
        assert len(msg.position) == 3, 'message should contain 3 joint values'

        # store current state
        self.cur_state_[0] = msg.position[0]
        self.cur_state_[1] = msg.position[1]
        self.cur_state_[2] = msg.position[2]

    def publish_effort(self):
        # error between current state and reference state
        cur_error = self.ref_state_ - self.cur_state_

        # control input that has proportional and derivate components
        u = self.kp_ * cur_error + self.kd_ * (cur_error - self.prev_error_) / self.update_rate_

        # publish control input
        msg = Float64MultiArray()
        msg.data.append(u[0,0])
        msg.data.append(u[1,0])
        msg.data.append(u[2,0])
        self.effort_publisher_.publish(msg)

        # store current error as previous error for next round
        self.prev_error_ = cur_error

def main(args=None):
    rclpy.init(args=args)
    kp = np.array([[0.1], [0.1], [50]])
    kd = np.array([[0.1], [0.1], [0.1]])
    pd_controller = PDControllerNode(1/30, kp, kd)
    rclpy.spin(pd_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
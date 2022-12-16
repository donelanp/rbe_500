# call service using: ros2 service call /EndEffectorVelRef interface_pkg/srv/EndEffectorVelRef "{velocity:[0,0,0,0,0,0]}"

import time
import array
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from interface_pkg.srv import EndEffectorVelRef, VelocityConversion
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class PlotTool():
    def __init__(self, update_period_):
        self.update_period = update_period_

        # initialize counters
        self.total_time = 10  # seconds
        self.total_counts = int(self.total_time / self.update_period)


    def newPlot(self, ref_state):
        self.ref_state = ref_state

        # reset counter
        self.counter = 0

        # total joint states stored over the time period
        self.cur_state_total = np.empty((3,0), float)

        # reference values for each time step will be the same
        q1_r_arr = np.full((1,self.total_counts), self.ref_state[0])
        q2_r_arr = np.full((1,self.total_counts), self.ref_state[1])
        q3_r_arr = np.full((1,self.total_counts), self.ref_state[2])
        self.ref_state_total = np.vstack((q1_r_arr, q2_r_arr, q3_r_arr))
        print("Starting plotting tool...")

    def new_state(self, cur_state_):
        self.cur_state_total = np.hstack([self.cur_state_total, cur_state_])
        
        # increment counter for each new position stored
        self.counter = self.counter + 1

        # if we are at the total time, print each joint's response
        if self.counter == self.total_counts:
            self.plot_joints()

    def plot_joints(self):
        x = np.linspace(0, self.total_time, self.total_counts)

        for i in range(3):
            plt.figure()
            y = self.cur_state_total[i]
            y_r = self.ref_state_total[i]
            plt.xlabel('Time (seconds)')
            plt.ylabel('Joint Velocity')
            plt.title('Joint {0} Response'.format(i+1))
            plt.plot(x,y, label="response")
            plt.plot(x,y_r, label="reference")
            plt.legend()
            plt.savefig('joint{0}_response.png'.format(i+1))

        print('Plots created...')

class MinimalClientAsync(Node):
    def __init__(self, service_name):
        super().__init__(service_name.lower() + '_client')
        self.client_ = self.create_client(VelocityConversion, service_name)
        while not self.client_.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('velocity conversion service not available, waiting again...')
        self.req_ = VelocityConversion.Request()

    def send_request(self, q, v):
        self.req_.joint_states = q
        self.req_.velocity = v
        self.future_ = self.client_.call_async(self.req_)
        rclpy.spin_until_future_complete(self, self.future_)
        return self.future_.result()

class Velocity_PDControllerNode(Node):
    def __init__(self, update_period, kp, kd):
        # ros node setup (service, client, subscriber, publisher, timer)
        super().__init__('velocity_pd_controller')
        self.reference_service_ = self.create_service(EndEffectorVelRef, 'EndEffectorVelRef', self.ref_callback)
        self.to_joint_vel_client_ = MinimalClientAsync('EndEffectorVelToJointVel')
        self.to_end_eff_vel_client_ = MinimalClientAsync('JointVelToEndEffectorVel')
        self.joint_subscriber_ = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.effort_publisher_ = self.create_publisher(Float64MultiArray, '/forward_effort_controller/commands', 10)
        self.timer_ = self.create_timer(update_period, self.publish_effort)

        # update rate for effort
        self.update_period_ = update_period

        # proportional gain (3x1 vector)
        self.kp_ = kp

        # derivative gain (3x1 vector)
        self.kd_ = kd

        # reference (goal) velocity of joints (3x1 vector)
        self.ref_vel_ = np.zeros((3,1))
        self.ref_vel_e = np.zeros((3,1))

        # current state of joints (3x1 vector)
        self.cur_state_ = np.zeros((3,1))

        # current velocity of joints (3x1 vector)
        self.cur_vel_ = np.zeros((3,1))

        # previous velocity error of joints (3x1 vector)
        self.prev_error_ = np.zeros((3,1))

        # create time response plotting tool
        self.plot_tool_ = PlotTool(self.update_period_)

        # create dummy plot that will be overwritten once a reference state is sent
        self.plot_tool_.newPlot(self.ref_vel_)


    # subscribes to joint states that are published by gazebo
    def joint_callback(self, msg):
        assert len(msg.position) == 3, 'message should contain 3 joint values'
        assert len(msg.velocity) == 3, 'message should contain 3 joint velocities'

        # store current state
        self.cur_state_ = np.expand_dims(np.array(msg.position), axis=1)

        # store current velocity
        self.cur_vel_ = np.expand_dims(np.array(msg.velocity), axis=1)

    # receives reference velocities from a ros2 service call...
    def ref_callback(self, request, response):
        #assert len(request.velocity) == 6, 'request should contain 6 end effector velocity values'
        # convert end effector velocity to joint velocity
        #   we need our joint positions to determine the velocity needed to go 
        #   in the direction of the end effector
        q = array.array('f', [self.cur_state_[0], self.cur_state_[1], self.cur_state_[2]])
        joint_vel = self.to_joint_vel_client_.send_request(q, request.velocity)

        # store reference joint velocity
        self.ref_vel_ = np.expand_dims(np.array(joint_vel.velocity), axis=1)
        self.ref_vel_e = np.asarray(request.velocity).reshape((3,1))

        # overwrite plotting tool for each new request
        self.plot_tool_.newPlot(self.ref_vel_)

        return response

    def publish_effort(self):
        # error between current velocity and reference velocity
        cur_error = self.ref_vel_ - self.cur_vel_

        # from the joint states, calculate the end effector position
        q = array.array('f', [self.cur_state_[0], self.cur_state_[1], self.cur_state_[2]])
        # convert cur_vel into a numpy array to pass into the service
        cur_vel_j = array.array('f', [self.cur_vel_[0], self.cur_vel_[1], self.cur_vel_[2]])
        # convert from joint velocities to end effector velocities
        self.cur_vel_e = self.to_end_eff_vel_client_.send_request(q, cur_vel_j).velocity 
        # reshape from a (3,) numpy array to a (3,1) numpy array
        self.cur_vel_e = np.asarray(self.cur_vel_e).reshape((3,1))


        cur_error = np.subtract(self.ref_vel_e, self.cur_vel_e)

        # control input that has proportional and derivate components
        u = self.kp_*cur_error + self.kd_*(cur_error-self.prev_error_) / self.update_period_
        
        # account for gravity
        u[2,0] = u[2,0] - 9.8

        # publish control input
        msg = Float64MultiArray()
        msg.data.append(u[0,0])
        msg.data.append(u[1,0])
        msg.data.append(u[2,0])
        self.effort_publisher_.publish(msg)

        # store current error as previous error for next round
        self.prev_error_ = cur_error

        # store current joint velocities
        self.plot_tool_.new_state(self.cur_vel_)


def main(args=None):
    rclpy.init(args=args)
    # kp[i] is the proportional gain for joint i+1
    kp = np.array([[0.5], [0.5], [0]])

    # kd[i] is the derivative gain for joint i+1
    kd = np.array([[0], [0], [0]])

    # create controller, update rate is 10 ms
    velocity_pd_controller = Velocity_PDControllerNode(0.01, kp, kd)

    rclpy.spin(velocity_pd_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

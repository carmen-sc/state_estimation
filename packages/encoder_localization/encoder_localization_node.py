#!/usr/bin/env python3
import numpy as np
import os
import rospy

import tf
import math

from fused_localization.srv import SpecialService, SpecialServiceResponse

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion


class LocalizationNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Localization Node
        Calculates the pose of the Duckiebot using only the wheel encoder data
        """

        # Initialize the DTROS parent class
        super(LocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
        self.radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 100)
        # distance between the wheels of the bot
        self.baseline = rospy.get_param(f'/{self.veh_name}/kinematics_node/baseline', 1)

        # define initial stuff
        self.x_pose = 0
        self.delta_dist_left = 0
        self.dist_left = 0
        self.delta_dist_right = 0
        self.dist_right = 0
        self.y_pose = 0
        self.theta = 0

        self.tol = 1/10000

        self.cb_dist_left = 0
        self.cb_dist_right = 0

        # define tf stuff
        self.br = tf.TransformBroadcaster()

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber(f'/{self.veh_name}/left_wheel_encoder_node/tick',
                                                       WheelEncoderStamped, self.calc_dist_left, queue_size=1)
        self.sub_encoder_ticks_right = rospy.Subscriber(f'/{self.veh_name}/right_wheel_encoder_node/tick',
                                                        WheelEncoderStamped, self.calc_dist_right, queue_size=1)

        # Publishers
        self.pub_baselink_pose = rospy.Publisher('~encoder_baselink_pose', TransformStamped, queue_size=1)

        self.service = rospy.Service('service', SpecialService, self.service_callback)

        self.log("Initialized")

        self.timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self.cb_baselink_pose)

    def cb_baselink_pose(self, timer_event):

        # get distances from the distance calculators
        self.delta_dist_left = self.dist_left - self.cb_dist_left
        self.delta_dist_right = self.dist_right - self.cb_dist_right
        delta_dist = self.delta_dist_right - self.delta_dist_left
        baselink_ark = (self.delta_dist_left + self.delta_dist_right) / 2

        # approximate theta
        delta_theta = delta_dist/self.baseline

        # calculate x and y in robot frame
        delta_x = math.cos(delta_theta) * baselink_ark
        delta_y = math.sin(delta_theta) * baselink_ark

        # calculate x and y in map frame
        self.x_pose = self.x_pose + math.cos(self.theta)*delta_x - math.sin(self.theta)*delta_y
        self.y_pose = self.y_pose + math.sin(self.theta)*delta_x + math.cos(self.theta)*delta_y
        self.theta = self.theta + delta_theta

        # translation and rotation arrays
        br_translation_array = (self.x_pose, self.y_pose, 0)
        br_rotation_array = tf.transformations.quaternion_from_euler(0, 0, self.theta)

        # publish & broadcast
        br_transform_msg = TransformStamped()
        br_transform_msg.header.stamp = timer_event.current_real
        br_transform_msg.header.frame_id = "map"
        br_transform_msg.child_frame_id = "encoder_baselink"
        br_transform_msg.transform.translation = Vector3(*br_translation_array)
        br_transform_msg.transform.rotation = Quaternion(*br_rotation_array)

        self.br.sendTransform(br_translation_array,
                              br_rotation_array,
                              br_transform_msg.header.stamp,
                              br_transform_msg.child_frame_id,
                              br_transform_msg.header.frame_id)

        self.pub_baselink_pose.publish(br_transform_msg)

        self.cb_dist_left = self.dist_left
        self.cb_dist_right = self.dist_right

    def calc_dist_left(self, msg):
        """ Update encoder distance information from ticks.
        """
        data = msg.data
        res = msg.resolution
        self.dist_left = 2 * 3.14159 * self.radius * data / res

    def calc_dist_right(self, msg):
        """ Update encoder distance information from ticks.
        """
        data = msg.data
        res = msg.resolution
        self.dist_right = 2 * 3.14159 * self.radius * data / res

    def service_callback(self, service_handle):
        """
            something with AT estimate fusion
        """
        transform = service_handle.transform.transform

        # reset deltas to zero
        self.delta_dist_left = 0
        self.dist_left = 0
        self.delta_dist_right = 0
        self.dist_right = 0

        # set AT estimate as initial estimate
        self.x_pose = transform.translation.x
        self.y_pose = transform.translation.y
        q_ = transform.rotation
        q = np.array([q_.x, q_.y, q_.z, q_.w])
        self.theta = tf.transformations.euler_from_quaternion(q, axes='sxyz')[2]

        return SpecialServiceResponse()



if __name__ == '__main__':
    node = LocalizationNode(node_name='encoder_localization_node')

    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("encoder_localization_node is up and running...")

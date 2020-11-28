#!/usr/bin/env python3

import numpy as np
import rospy
import tf
from fused_localization.srv import SpecialService

from duckietown.dtros import DTROS, NodeType

from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion


class LocalizationNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Localization Node
        Calculates the pose of the Duckiebot using only AprilTags
        """

        # Initialize the DTROS parent class
        super(LocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")

        # Classes
        self.br = tf.TransformBroadcaster()

        # Subscribers
        self.at_sub = rospy.Subscriber('~at_baselink_pose', TransformStamped, self.at_callback)
        self.encoder_sub = rospy.Subscriber('~encoder_baselink_pose', TransformStamped, self.encoder_callback)

        # Publishers
        self.pub_baselink_pose = rospy.Publisher('~fused_baselink_pose', TransformStamped, queue_size=1)

        # Services
        self.service = rospy.ServiceProxy('service', SpecialService)

        self.log("Initialized")

    def at_callback(self, msg):
        """
            Maps map --> at_baselink and sends it to the encoder node
        """
        self.service(msg)

    def encoder_callback(self, msg):
        self.pub_baselink_pose.publish(msg)

        transform_msg = TransformStamped()
        transform_msg.header.stamp = msg.header.stamp
        transform_msg.header.frame_id = "map"
        transform_msg.child_frame_id = "fused_baselink"
        translation_array = np.array(
            [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        rotation_array = np.array(
            [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])

        self.br.sendTransform(translation_array,
                              rotation_array,
                              transform_msg.header.stamp,
                              transform_msg.child_frame_id,
                              transform_msg.header.frame_id)

    def onShutdown(self):
        super(LocalizationNode, self).onShutdown()


if __name__ == '__main__':
    node = LocalizationNode(node_name='fused_localization_node')

    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("encoder_localization_node is up and running...")

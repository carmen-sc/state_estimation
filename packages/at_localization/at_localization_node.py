#!/usr/bin/env python3
import numpy as np
import os
import rospy
import cv2
import tf
import tf2_ros
import math
import copy
import yaml
from cv_bridge import CvBridge, CvBridgeError

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from dt_apriltags import Detector
from augmented_reality_helper_module import Helpers

from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion


class LocalizationNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Localization Node
        Calculates the pose of the Duckiebot using only AprilTags
        """

        # Initialize the DTROS parent class
        super(LocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")

        # define initial stuff
        self.camera_x = 0.065
        self.camera_z = 0.11
        self.camera_slant = 19/180*math.pi
        self.height_streetsigns = 0.07

        # Load calibration files
        self._res_w = 640
        self._res_h = 480
        self.calibration_call()

        # define tf stuff
        self.br = tf.TransformBroadcaster()
        self.static_tf_br = tf2_ros.StaticTransformBroadcaster()

        # Initialize classes
        self.at_detec = Detector(searchpath=['apriltags'], families='tag36h11', nthreads=4, quad_decimate=4.0,
                                 quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
        self.helper = Helpers(self.current_camera_info)
        self.bridge = CvBridge()

        # Subscribers
        self.cam_image = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage,
                                          self.callback, queue_size=10)

        # Publishers
        self.pub_baselink_pose = rospy.Publisher('~at_baselink_pose', TransformStamped, queue_size=1)

        # static broadcasters
        self.broadcast_static_transform_baselink()

        self.log("Initialized")

    def callback(self, image):
        """
            Callback method that ties everything together
        """
        # convert the image to cv2 format
        image = self.bridge.compressed_imgmsg_to_cv2(image)

        # undistort the image
        image = self.helper.process_image(image)

        # detect AprilTags
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.at_detec.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

        for tag in tags:
            self.broadcast_tag_camera_transform(tag)

    def broadcast_tag_camera_transform(self, tag):
        """
            compute the transform between AprilTag and camera
        """
        camera_to_tag_R = tag.pose_R
        camera_to_tag_t = tag.pose_t

        # build T matrix
        camera_to_tag_T = np.append(camera_to_tag_R, camera_to_tag_t, axis=1)
        camera_to_tag_T = np.append(camera_to_tag_T, [[0, 0, 0, 1]], axis=0)

        # translate T into the rviz frame convention
        T_rviz = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        self.rviz_to_tag_T = np.matmul(T_rviz, camera_to_tag_T)

        # translation and rotation arrays
        br_translation_array = np.array(self.rviz_to_tag_T[0:3, 3])
        br_rotation_array = tf.transformations.quaternion_from_matrix(self.rviz_to_tag_T)

        # publish & broadcast
        br_transform_msg = self.broadcast_packer("camera", "apriltag", br_translation_array, br_rotation_array)

        self.br.sendTransform(br_translation_array,
                              br_rotation_array,
                              br_transform_msg.header.stamp,
                              br_transform_msg.child_frame_id,
                              br_transform_msg.header.frame_id)

        # self.pub_baselink_pose.publish(br_transform_msg)
        self.broadcast_static_frame_rotation_transform()
        self.broadcast_static_transform_apriltag()
        self.broadcast_map_baselink_transform()

    def broadcast_map_baselink_transform(self):
        # baselink --> map transform tf_source_end
        map_baselink_T = np.linalg.inv(self.camera_baselink_T @ self.rviz_to_tag_T @ self.frame_rotation_T @ self.map_tag_T)
        # map_baselink_T = self.map_tag_T @ self.rviz_to_tag_T @ self.camera_baselink_T
        br_translation_array = np.array(map_baselink_T[0:3, 3])
        br_rotation_array = tf.transformations.quaternion_from_matrix(map_baselink_T)
        br_transform_msg = self.broadcast_packer("at_baselink", "map", br_translation_array, br_rotation_array)

        self.pub_baselink_pose.publish(br_transform_msg)

    def broadcast_static_transform_baselink(self):
        # camera --> baselink transform
        static_translation_array = (self.camera_x, 0, self.camera_z)
        rotation_matrix = np.array([[math.cos(self.camera_slant), 0, math.sin(self.camera_slant), 0], [0, 1, 0, 0],
                                        [-math.sin(self.camera_slant), 0, math.cos(self.camera_slant), 0], [0, 0, 0, 1]])
        static_rotation_array = tf.transformations.quaternion_from_matrix(rotation_matrix)
        self.camera_baselink_T = np.copy(rotation_matrix)
        self.camera_baselink_T[0:3, 3] = static_translation_array
        static_transform_msg = self.broadcast_packer("at_baselink", "camera", static_translation_array, static_rotation_array)

        # broadcast packed message
        self.static_tf_br.sendTransform(static_transform_msg)

    def broadcast_static_frame_rotation_transform(self):
        # change in coordinate frame from ATD convention to rviz convention
        static_translation_array = (0, 0, 0)
        static_rotation = np.linalg.inv(np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        self.frame_rotation_T = np.copy(static_rotation)
        self.frame_rotation_T[0:3, 3] = static_translation_array
        static_rotation_array = tf.transformations.quaternion_from_matrix(static_rotation)
        static_transform_msg = self.broadcast_packer("apriltag", "apriltag_rot", static_translation_array, static_rotation_array)

        # broadcast packed message
        self.static_tf_br.sendTransform(static_transform_msg)

    def broadcast_static_transform_apriltag(self):
        # map --> AprilTag transform
        static_translation_array = (0, 0, -self.height_streetsigns)
        static_rotation_array = tf.transformations.quaternion_from_matrix(np.eye(4))
        self.map_tag_T = np.copy(np.eye(4))
        self.map_tag_T[0:3, 3] = static_translation_array
        static_transform_msg = self.broadcast_packer("apriltag_rot", "map", static_translation_array, static_rotation_array)
        # broadcast packed message
        self.static_tf_br.sendTransform(static_transform_msg)

    def calibration_call(self):
        # copied from the camera driver (dt-duckiebot-interface)

        # For intrinsic calibration
        self.cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        self.frame_id = rospy.get_namespace().strip('/') + '/camera_optical_frame'
        self.cali_file = self.cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (self.cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        self.original_camera_info = self.load_camera_info(self.cali_file)
        self.original_camera_info.header.frame_id = self.frame_id
        self.current_camera_info = copy.deepcopy(self.original_camera_info)
        self.update_camera_params()
        self.log("Using calibration file: %s" % self.cali_file)

        # extrinsic calibration
        self.ex_cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        self.ex_cali_file = self.ex_cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"
        self.ex_cali = self.readYamlFile(self.ex_cali_file)

        # define camera parameters for AT detector
        self.camera_params = [self.current_camera_info.K[0], self.current_camera_info.K[4], self.current_camera_info.K[2], self.current_camera_info.K[5]]

    def update_camera_params(self):
        # copied from the camera driver (dt-duckiebot-interface)
        """ Update the camera parameters based on the current resolution.
        The camera matrix, rectification matrix, and projection matrix depend on
        the resolution of the image.
        As the calibration has been done at a specific resolution, these matrices need
        to be adjusted if a different resolution is being used.
        """

        scale_width = float(self._res_w) / self.original_camera_info.width
        scale_height = float(self._res_h) / self.original_camera_info.height

        scale_matrix = np.ones(9)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[4] *= scale_height
        scale_matrix[5] *= scale_height

        # Adjust the camera matrix resolution
        self.current_camera_info.height = self._res_h
        self.current_camera_info.width = self._res_w

        # Adjust the K matrix
        self.current_camera_info.K = np.array(self.original_camera_info.K) * scale_matrix

        # Adjust the P matrix (done by Rohit)
        scale_matrix = np.ones(12)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[5] *= scale_height
        scale_matrix[6] *= scale_height
        self.current_camera_info.P = np.array(self.original_camera_info.P) * scale_matrix

    def readYamlFile(self, fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    @staticmethod
    def load_camera_info(filename):
        # copied from the camera driver (dt-duckiebot-interface)
        """Loads the camera calibration files.
        Loads the intrinsic camera calibration.
        Args:
            filename (:obj:`str`): filename of calibration files.
        Returns:
            :obj:`CameraInfo`: a CameraInfo message object
        """
        with open(filename, 'r') as stream:
            calib_data = yaml.load(stream)
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info

    @staticmethod
    def broadcast_packer(header_frame, child_frame, translation_array, rotation_array):
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = header_frame
        transform_msg.child_frame_id = child_frame
        transform_msg.transform.translation = Vector3(*translation_array)
        transform_msg.transform.rotation = Quaternion(*rotation_array)
        return transform_msg

    def onShutdown(self):
        super(LocalizationNode, self).onShutdown()


if __name__ == '__main__':
    node = LocalizationNode(node_name='at_localization_node')

    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("encoder_localization_node is up and running...")

#!/usr/bin/env python3
# ============================================================
# aruco_detector_node.py
# Subscribes to the wrist Kinect RGB image stream,
# detects ArUco tags, and publishes the pose of each
# detected instrument as a separate topic.
#
# Dependencies: cv_bridge, opencv, sensor_msgs, geometry_msgs
# ============================================================

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

# ---- INSTRUMENT MAP -----------------------------------------
# Loaded from param server (set by launch file from instruments.yaml)
# Fallback hardcoded here in case params not loaded, can add more instruments
DEFAULT_INSTRUMENT_MAP = {
    0: "scalpel",
    1: "tissue_forceps",
    2: "curved_hemostat",
    3: "needle_holder",
    4: "kelly_hemostat",
    5: "scissors",
}

# ---- NODE CLASS ---------------------------------------------
# Using a class keeps all state organized
# instead of scattered global variables

class ArucoDetectorNode:

    def __init__(self):
        rospy.init_node('aruco_detector_node', anonymous=False)
        rospy.loginfo("ArUco detector node starting...")

        # --- Load parameters from ROS param server ---
        # These come from instruments.yaml loaded by the launch file
        self.instrument_map = rospy.get_param(
            '~instrument_map', DEFAULT_INSTRUMENT_MAP
        )
        # Convert string keys to int (YAML loads them as strings)
        self.instrument_map = {
            int(k): v for k, v in self.instrument_map.items()
        }

        self.rgb_topic = rospy.get_param(
            '~kinect_rgb_topic', '/rgb/image_raw'
        )
        self.tag_size = rospy.get_param('~tag_size_meters', 0.05)

        # --- ArUco setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )

        # --- Camera calibration ---
        # Needed for 3D pose estimation from 2D image
        # Will be populated when we receive CameraInfo message
        self.camera_matrix = None
        self.dist_coeffs = None

        # --- cv_bridge ---
        # Converts ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # --- Publishers ---
        # One pose topic per instrument (Option B from earlier discussion)
        self.pose_pubs = {}
        for tag_id, name in self.instrument_map.items():
            topic = f'/tray/{name}_pose'
            self.pose_pubs[tag_id] = rospy.Publisher(
                topic, PoseStamped, queue_size=1
            )
            rospy.loginfo(f"Publishing {name} pose to {topic}")

        # Also publish which instruments are currently visible
        self.visible_pub = rospy.Publisher(
            '/tray/visible_instruments', String, queue_size=1
        )

        # --- Subscribers ---
        # Camera calibration (needed once for pose estimation)
        rospy.Subscriber(
            self.rgb_topic.replace('image_raw', 'camera_info'),
            CameraInfo,
            self.camera_info_callback
        )

        # Main image stream
        rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.image_callback
        )

        rospy.loginfo(f"Subscribing to {self.rgb_topic}")
        rospy.loginfo("ArUco detector ready.")

    def camera_info_callback(self, msg):
        """
        Receives camera calibration data once.
        Needed to compute real 3D positions from 2D detections.
        After receiving once, we unsubscribe.
        """
        if self.camera_matrix is None:
            # Reshape the flat array into 3x3 matrix
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera calibration received.")

    def image_callback(self, msg):
        """
        Runs on every incoming camera frame.
        Detects all ArUco tags and publishes poses.
        """
        # Skip if we don't have camera calibration yet
        if self.camera_matrix is None:
            return

        # Convert ROS Image → OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all ArUco tags in the frame
        corners, ids, _ = self.detector.detectMarkers(gray)

        visible_instruments = []

        if ids is not None:
            # Estimate 3D pose of each detected tag
            # rvecs = rotation vectors, tvecs = translation vectors
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.tag_size,        # physical size of tag in meters
                self.camera_matrix,   # camera intrinsics
                self.dist_coeffs      # lens distortion
            )

            for i, tag_id in enumerate(ids.flatten()):
                tag_id = int(tag_id)

                if tag_id not in self.instrument_map:
                    continue

                instrument_name = self.instrument_map[tag_id]
                visible_instruments.append(instrument_name)

                # tvecs[i][0] = [x, y, z] position in camera frame (meters)
                tx, ty, tz = tvecs[i][0]

                # Build ROS PoseStamped message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                # Frame this pose is relative to (camera frame)
                # TF will convert to robot base frame later
                pose_msg.header.frame_id = "oarbot_blue_kinect_rgb_camera_link"
                pose_msg.pose.position.x = tx
                pose_msg.pose.position.y = ty
                pose_msg.pose.position.z = tz
                # Orientation from rotation vector
                pose_msg.pose.orientation.w = 1.0  # simplified for now

                # Publish this instrument's pose
                self.pose_pubs[tag_id].publish(pose_msg)

                rospy.loginfo_throttle(
                    1.0,  # only log once per second max
                    f"Detected {instrument_name} at "
                    f"x={tx:.3f} y={ty:.3f} z={tz:.3f}m"
                )

        # Publish list of currently visible instruments
        if visible_instruments:
            self.visible_pub.publish(
                String(", ".join(visible_instruments))
            )


def main():
    node = ArucoDetectorNode()
    rospy.spin()  # keep node alive, callbacks handle everything


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

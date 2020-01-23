# !/usr/bin/env python
import numpy as np
import rosgraph
import tf
import rospy
import tf.transformations as T

from geometry_msgs.msg import *


class UserPerspectiveTransformation(object):

    def __init__(self, using_ros, user_pose, rotation_type, user_id,
                 perspective_matrix=np.eye(3), robot_pose=Pose()):

        self._user_angle = 0.0
        self._user_pose = user_pose
        self._rotation_type = rotation_type
        self._using_ros = using_ros
        self._user_id = user_id
        self._perspective_matrix = perspective_matrix
        self._robot_pose = robot_pose

        if rosgraph.is_master_online():
            self._ros_active = True
            if self._using_ros:
                self._tf_listener = tf.TransformListener()
            else:
                self._tf_listener = None
        else:
            self._ros_active = False

    def updatePos(self, user_pos):
        self._user_pose.position = user_pos

    def updateRotation(self, user_orientation):
        self._user_pose.orientation = user_orientation

    def updatePose(self, user_pose):
        self._user_pose = user_pose

    def perspectiveTransformation(self, orig_point=Point()):

        if self._ros_active and self._using_ros:

            try:
                # Create TF message for point
                point_robot = PointStamped()
                point_robot.header.stamp = rospy.Time(0)
                point_robot.header.frame_id = '/base_link'
                point_robot.point.x = orig_point.x
                point_robot.point.y = orig_point.y
                point_robot.point.z = orig_point.z

                # Transform point from robot space to user space using TFs
                point_world = self._tf_listener.transformPoint('/map', point_robot)
                point_user_tf = self._tf_listener.transformPoint('/' + str(self._user_id), point_world)

                # Apply 2D perspective transformation
                point_user = np.array([point_user_tf.point.x, point_user_tf.point.y, point_user_tf.point.z])
                user_point_perspective = self._perspective_matrix.dot(point_user)
                return user_point_perspective

            except (ValueError, rospy.ROSSerializationException, tf.LookupException,
                    tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr('[PERSPECTIVE TRANSFORMATION]: Caught Error: %s' % e)
                return None

        else:

            # Get robot and user orientation transformations under Euler Angles
            if self._rotation_type.find('euler') != -1:

                user_euler = T.euler_from_quaternion((self._user_pose.orientation.x, self._user_pose.orientation.y,
                                                      self._user_pose.orientation.z, self._user_pose.orientation.w))
                robot_euler = T.euler_from_quaternion((self._robot_pose.orientation.x, self._robot_pose.orientation.y,
                                                       self._robot_pose.orientation.z, self._robot_pose.orientation.w))

                user_transformation = T.euler_matrix(user_euler[0], user_euler[1], user_euler[2])

                robot_transformation = T.euler_matrix(robot_euler[0], robot_euler[1], robot_euler[2])

            # Get robot and user orientation transformations under Quaternions
            elif self._rotation_type.find('quaternion') != -1:

                user_transformation = T.quaternion_matrix((self._user_pose.orientation.x,
                                                           self._user_pose.orientation.y,
                                                           self._user_pose.orientation.z,
                                                           self._user_pose.orientation.w))
                robot_transformation = T.quaternion_matrix((self._robot_pose.orientation.x,
                                                            self._robot_pose.orientation.y,
                                                            self._robot_pose.orientation.z,
                                                            self._robot_pose.orientation.w))

            else:
                print('Invalid rotation type, impossible to transform points')
                return None

            # Add translation of user and robot to transformation matrix
            robot_transformation[0, 3] = self._robot_pose.position.x
            robot_transformation[1, 3] = self._robot_pose.position.y
            robot_transformation[2, 3] = self._robot_pose.position.z

            user_transformation[0, 3] = self._user_pose.position.x
            user_transformation[1, 3] = self._user_pose.position.y
            user_transformation[2, 3] = self._user_pose.position.z

            # Transform point from robot space to user space
            robot_transformation = np.linalg.inv(robot_transformation)
            point_robot = np.array([[orig_point.x], [orig_point.y], [orig_point.z], [1]])
            point_world = robot_transformation.dot(point_robot)

            # Apply 2D perspective transformation
            point_user = user_transformation.dot(point_world)
            user_point_perspective = self._perspective_matrix.dot(point_user)
            return user_point_perspective

    def getPerspectiveTrajectory(self, orig_trajectory):

        transformed_trajectory = []

        # compute perspective transformation
        for i in range(len(orig_trajectory)):

            user_ptr = self.perspectiveTransformation(orig_trajectory[i])
            transformed_trajectory += [Point(user_ptr[0], user_ptr[1], user_ptr[2])]

        return transformed_trajectory

# !/usr/bin/env python
import numpy as np
import math
import roboticstoolbox as rtb
import tf.transformations as T
import spatialmath as sm

from termcolor import colored
from geometry_msgs.msg import Pose, Quaternion, Point


def get_forward_kinematics(robot_model: rtb.DHRobot):

    fk_mapping = {
        'IRB4600-40': forward_kinematics_irb4600
    }

    if robot_model.name in fk_mapping.keys():
        return fk_mapping[robot_model.name]
    else:
        print(colored('Model %s not recognized, returning general forward kinematics method' % robot_model.name))
        return forward_kinematics


def get_dict_keys(d: dict) -> list:
    return list(d.keys())


def get_dict_values(d: dict) -> list:
    return list(d.values())


def forward_kinematics(robot_model: rtb.DHRobot, joint_trajectory: np.ndarray) -> np.ndarray:

    cartesian_trajectory = []

    for i in range(len(joint_trajectory)):

        fk_result = robot_model.fkine(joint_trajectory[i])
        cartesian_trajectory += [fk_result.t * 1000]        # need the *1000 to convert from meters to milimeters

    return np.array(cartesian_trajectory)


def forward_kinematics_irb4600(robot_model: rtb.DHRobot, joint_trajectory: np.ndarray) -> np.ndarray:

    cartesian_trajectory = []

    for i in range(len(joint_trajectory)):

        fk_result = robot_model.fkine(joint_trajectory[i])  # get the end-effector in relation to robot's base

        # transform position to robot space
        transformation = np.array([[0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]])
        robot_fk = np.linalg.inv(transformation).dot(np.concatenate((fk_result.t, 1), axis=None))[:-1]
        cartesian_trajectory += [robot_fk * 1000]           # need the *1000 to convert from meters to milimeters

    return np.array(cartesian_trajectory)


def cartesian_to_joint_optimization(robot_model: rtb.DHRobot, cartesian_optimization: np.ndarray,
                                    stored_trajectory: np.ndarray) -> np.ndarray:

    if len(cartesian_optimization) != len(stored_trajectory):
        print(colored('Old trajectory length different from update optimization. Impossible cartesian-joint conversion',
                      'red'))
        return np.array([])

    joint_optimization = []
    for i in range(len(cartesian_optimization)):

        jacobian = robot_model.jacobe()
        full_optimization = np.concatenate((cartesian_optimization[i], np.zeros(3)), axis=None) # no orientation
        joint_optimization += [np.linalg.pinv(jacobian).dot(full_optimization)]                 # rrmc application

    return np.array(joint_optimization)


def transform_world(trajectory: np.ndarray, pose: Pose) -> np.ndarray:
    orientation = Quaternion(pose.orientation[0], pose.orientation[1],
                             pose.orientation[2], pose.orientation[3])
    position = Point(pose.position[0], pose.position[1], pose.position[2])

    euler = T.euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))

    transformation = T.euler_matrix(euler[0], euler[1], euler[2])

    transformation[0, 3] = position.x
    transformation[1, 3] = position.y
    transformation[2, 3] = position.z

    transformed_trajectory = []
    for i in range(len(trajectory)):
        projected_point = transformation.dot(np.concatenate((trajectory[i], 1), axis=None))[:-1]
        transformed_trajectory += [projected_point]

    return np.array(transformed_trajectory)


def prepare_trajectory(trajectory: np.ndarray, robot_pose: Pose, robot_model: rtb.DHRobot,
                       model_pose: np.ndarray) -> np.ndarray:

    robot_orientation = Quaternion(robot_pose.orientation[0], robot_pose.orientation[1],
                                   robot_pose.orientation[2], robot_pose.orientation[3])
    robot_position = Point(robot_pose.position[0], robot_pose.position[1], robot_pose.position[2])

    robot_euler = T.euler_from_quaternion((robot_orientation.x, robot_orientation.y,
                                           robot_orientation.z, robot_orientation.w))

    robot_transformation = T.euler_matrix(robot_euler[0], robot_euler[1], robot_euler[2])

    robot_transformation[0, 3] = robot_position.x
    robot_transformation[1, 3] = robot_position.y
    robot_transformation[2, 3] = robot_position.z
    robot_transformation = np.linalg.inv(robot_transformation)

    transformed_trajectory = []
    for i in range(len(trajectory)):
        projected_point = robot_transformation.dot(np.concatenate((trajectory[i], 1), axis=None)) / 1000
        projected_point = model_pose.dot(projected_point)[:-1]
        ik_result, fail, err = robot_model.ikine(sm.SE3(projected_point[0], projected_point[1], projected_point[2]))
        transformed_trajectory += [ik_result]

    return np.array(transformed_trajectory)



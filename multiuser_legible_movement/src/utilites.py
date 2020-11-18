# !/usr/bin/env python
import numpy as np
import math
import roboticstoolbox as rtb

from termcolor import colored


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

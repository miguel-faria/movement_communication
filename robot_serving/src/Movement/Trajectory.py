"""
    TRAJECTORY CLASS

    This class has all the information needed to create
    and send a trajectory for the Baxter robot over ROS.

    The main methods in this class are:
        -   add_point(position, times) : adds a point and respective time to the movement trajectory
        -   wait(timeout) : waits at most the timeout time for the robot to complete trajectory
        -   initialize_limb(limb) : initializes the joint names for the corresponding limb in the '_goal' message
        -   start() : sends the message, signaling Baxter to start the movement
        -   stop() : sends Baxter a signal, making it stop the trajectory execution
        -   result() : gets feedback about the trajectory execution
        -   start_movement(target) : starts the movement towards the given 'target' or if not target is given uses the
        goal already stored
"""

import sys
from copy import copy

import actionlib
import rospy
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)


class Trajectory(object):

    SETUP_MAX_TIME = 10
    TARGET_DIM = 2

    """
        Constructor
    """
    def __init__(self, limb):
        self._limb = limb
        ns = '/robot/_limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction
        )
        self._goal = FollowJointTrajectoryGoal()
        server_up = self._client.wait_for_server(timeout=rospy.Duration(self.SETUP_MAX_TIME))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.initialize_limb(limb)

    """
        Getters and Setters
    """
    @property
    def limb(self):
        return self._limb

    @limb.setter
    def limb(self, value):
        self._limb = value

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        # case the value is a sequence of time steps and positions
        if len(value) == self.TARGET_DIM & len(value[0]) > 1:
            if len(value[0]) == len(value[1]):
                new_traj = FollowJointTrajectoryGoal()
                for i in range(self.TARGET_DIM):
                    point = JointTrajectoryPoint()
                    point.positions = copy(value[1][i])
                    point.time_from_start = rospy.Duration(value[0][i])
                    new_traj.trajectory.points.append(point)
                self._goal = new_traj
            else:
                print('Must have the same number of time steps as joint positions')
        # case it is an already complete goal just set
        else:
            self._goal = value

    @property
    def client(self):
        return self._client

    def is_goal_set(self):
        return self._goal != []

    """
        Modifiers
    """
    def add_point(self, position, time):
        point = JointTrajectoryPoint()
        point.positions = copy(position)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def wait(self, max_time=15):
        self._client.wait_for_result(timeout=rospy.Duration(max_time))

    def initialize_limb(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in
                                             ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def result(self):
        return self._client.get_result()

    def movement_state(self):
        return self._client.get_state()

    def start_movement(self, target=None):
        if target is None:
            if not self._goal:
                rospy.logerr("Can not move!! No goal specified.")
            else:
                self.start()
        elif len(target) == self.TARGET_DIM & (len(target[0]) == len(target[1])):
            if self._goal.trajectory.points:
                self._goal = FollowJointTrajectoryGoal()
            for i in range(self.TARGET_DIM):
                self.add_point(target[1][i], target[0][i])
            self.start()

        else:
            rospy.logerr("Invalid call to \'start_movement\', recheck parameters.")

#!/usr/bin/env python

from actionlib_msgs.msg import *
import rospy

from Baxter_Movement_Aux.Trajectory import *
from robot_serving.srv import *


class BaxterConnection(object):

    """
        Class Variables and Constants
    """

    CANCEL_MOVEMENT = 1
    RESTART_MOVEMENT = 2

    baxter_arm_control = {}

    def __init__(self):
        print("Starting Connection to Baxter")
        self._baxter_arm_control = {'left': Trajectory('left'), 'right': Trajectory('right')}
        self._start_trajectory_service = rospy.Service('baxter_start_trajectory', RobotMovementSendTrajectory,
                                                       self.handle_trajectory_start)
        self._movement_feedback_service = rospy.Service('baxter_movement_feedback', RobotMovementFeedback,
                                                        self.handle_trajectory_feedback)
        self._stop_start_trajectory_service = rospy.Service('baxter_stop_start_trajectory', RobotMovementCancelTrajectory,
                                                            self.handle_trajectory_stop_start_movement)

        print("Services ready!")

    """
        Handle for the service of starting a new movement.
        The service is described in file RobotMovementSendTrajectory.srv:
            limb - refers to which limb the movement is to be applied
            robot_trajectory - the sequence of values for each joint at each time step
            ---
            movement_status - an integer with the code for the status of the current movement
    """
    def handle_trajectory_start(self, req):
        traj = self._baxter_arm_control[req.limb]
        if not traj.is_goal_set():
            traj.start_movement(req.robot_trajectory)
        return RobotMovementSendTrajectoryResponse(traj.movement_state())

    """
        Handle for the service of getting feedback for a movement.
        The service is described in file RobotMovementFeedback.srv:
            limb - refers to which limb the movement is to be applied
            ---
            feedback - the state at which the current movement, of the given limb, is
    """
    def handle_trajectory_feedback(self, req):
        traj = self._baxter_arm_control[req.limb]
        if traj.movement_state() in \
                [GoalStatus.PREEMPTED, GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.RECALLED, GoalStatus.REJECTED]:
            return RobotMovementFeedbackResponse(traj.result())
        else:
            return RobotMovementFeedbackResponse(traj.movement_state())

    """
        Handle for the service of stopping the current movement and optionally starting a new one
        The service is described in file RobotMovementCancelTrajectory.srv:
            limb - refers to which limb the movement is to be applied
            service_code - code that encodes whether this service only cancels the movement or if it starts a new one
            robot_trajectory - an optional field with the new trajectory for the robot.
            ---
            movement_status - an integer with the code for the status of the current movement
    """
    def handle_trajectory_stop_start_movement(self, req):
        traj = self._baxter_arm_control[req.limb]
        traj.stop_all()
        traj.wait(2.5)
        traj.send_neutral()
        traj.wait(5)
        if req.service_code == self.CANCEL_MOVEMENT:
            return RobotMovementCancelTrajectoryResponse(GoalStatus.ABORTED)
        elif req.service_code == self.RESTART_MOVEMENT:
            traj.start_movement(req.robot_trajectory)
            return RobotMovementCancelTrajectoryResponse(traj.movement_state())

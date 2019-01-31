#!/usr/bin/env python

import random
import rospy

from actionlib_msgs.msg import *

from robot_serving.srv import *

"""
    Global Variables and Constants
"""
CANCEL_MOVEMENT = 1
RESTART_MOVEMENT = 2


def handle_trajectory_start(req):
    print("Status = " + str(GoalStatus.PENDING))
    return RobotMovementSendTrajectoryResponse(GoalStatus.PENDING)


def handle_trajectory_feedback(req):
    response = GoalStatus.ACTIVE if random.random() > 0.75 else GoalStatus.SUCCEEDED
    print("Status = " + str(response))
    return RobotMovementFeedbackResponse(response)


def handle_trajectory_stop_start_movement(req):
    if req.service_code == CANCEL_MOVEMENT:
        print("Cancel message")
        return RobotMovementCancelTrajectoryResponse(GoalStatus.ABORTED)
    elif req.service_code == RESTART_MOVEMENT:
        print("Restart Message")
        return RobotMovementCancelTrajectoryResponse(GoalStatus.ACTIVE)


def baxter_connection_service():
    print("Setting up Baxter Connection Service")
    rospy.init_node('baxter_connection_server')
    start_trajectory_service = rospy.Service('baxter_start_trajectory', RobotMovementSendTrajectory, handle_trajectory_start)
    movement_feedback_service = rospy.Service('baxter_movement_feedback', RobotMovementFeedback, handle_trajectory_feedback)
    stop_start_trajectory_service = rospy.Service('baxter_stop_start_trajectory', RobotMovementCancelTrajectory,
                                                  handle_trajectory_stop_start_movement)
    print("Service server ready!")
    rospy.spin()

if __name__ == "__main__":
    baxter_connection_service()

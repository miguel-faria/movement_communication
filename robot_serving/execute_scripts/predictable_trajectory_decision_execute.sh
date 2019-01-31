#!/usr/bin/env bash
cd ~/catkin_ws/src/robot_serving/
source ./execute_scripts/variable_def.sh

echo "----------------------------------------------"
echo "-- Predictable Trajectory Movement Decision --"
echo "----------------------------------------------"
#gnome-terminal -e "rosrun baxter_interface joint_trajectory_action_server.py"
#gnome-terminal -e "python $BaxterInteractionServer"
gnome-terminal -e "python $BaxterMovementServer -t predict"
$MovementExecutableSimple $CupSubTopic "predictable" $PredictableTrajSrv $StartMovSrv $FeedbackSrv $RestartMovSrv $BaxterArm $DecisionPeriod

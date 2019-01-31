#!/usr/bin/env bash
cd ~/catkin_ws/src/robot_serving/
source ./execute_scripts/variable_def.sh

echo "------------------------------------"
echo "- Launching Vision Processing Node -"
echo "------------------------------------"
#gnome-terminal -e "roslaunch kinect2_bridge kinect2_bridge.launch depth_method:='opencl' reg_method:='opencl'"
$VisionProcessing_Executable $ColorTopic $DepthTopic $CupPosTopic $CalibParams
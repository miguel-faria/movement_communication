#!/usr/bin/env bash
cd ~/catkin_ws/src/robot_serving/
source ./execute_scripts/variable_def.sh

echo "------------------------"
echo "-- Interaction Server --"
echo "------------------------"

python $BaxterInteractionServer $1 $2
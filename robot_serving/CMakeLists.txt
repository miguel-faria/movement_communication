cmake_minimum_required(VERSION 2.8.3)
project(robot_serving)
FILE(GLOB_RECURSE COLOR_SEGMENTATION_SRC_FILES src/Vision/Color\ Segmentation/*.cpp)
FILE(GLOB_RECURSE MOVEMENT_MANAGER_SRC_FILES src/Movement/Manager_Aux/*.cpp src/Movement/Trajectory_Types/*.cpp) 

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  std_srvs
  message_generation
  image_transport
  sensor_msgs
  cv_bridge
)

find_package( OpenCV REQUIRED )

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Cups.msg
  FollowBehaviour.msg
  Follow_mode.msg
  PerceivedFace.msg
  PMPPoint.msg
  PMPTraj.msg
  screen_mode.msg
  ManageExpression.msg
  SpeechCues.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  Movement.srv
  RobotMovementCancelTrajectory.srv
  RobotMovementFeedback.srv
  RobotMovementSendTrajectory.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  std_srvs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES Thesis
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime std_srvs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ./include
#  /home/miguel/catkin_ws/src/thesis/devel/include
)

add_executable( vision_processing_color_segmentation src/Vision/vision_processing_color_segmentation.cpp ${COLOR_SEGMENTATION_SRC_FILES} )
target_compile_features(vision_processing_color_segmentation PRIVATE cxx_range_for)
add_dependencies(vision_processing_color_segmentation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vision_processing_color_segmentation ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable( movement_manager_single_trajectory src/Movement/manager_single_trajectory.cpp ${MOVEMENT_MANAGER_SRC_FILES} )
target_compile_features(movement_manager_single_trajectory PRIVATE cxx_range_for)
add_dependencies(movement_manager_single_trajectory ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(movement_manager_single_trajectory ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable( movement_manager_mixed_trajectory src/Movement/manager_mixed_trajectory.cpp ${MOVEMENT_MANAGER_SRC_FILES} )
target_compile_features(movement_manager_mixed_trajectory PRIVATE cxx_range_for)
add_dependencies(movement_manager_mixed_trajectory ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(movement_manager_mixed_trajectory ${OpenCV_LIBS} ${catkin_LIBRARIES})

######################################
#### FOR TRAJECTORY RECORDING!!!! ####
######################################
add_executable( cup_identification_recording trajectory_recording/cup_identification_recording.cpp ${COLOR_SEGMENTATION_SRC_FILES} )
target_compile_features(cup_identification_recording PRIVATE cxx_range_for)
add_dependencies(cup_identification_recording ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cup_identification_recording ${OpenCV_LIBS} ${catkin_LIBRARIES})


##############################
#### FOR TESTING ONLY!!!! ####
##############################
add_executable( testing_single_trajectory_decision src/test_src/testing_single_trajectory_decision.cpp ${MOVEMENT_MANAGER_SRC_FILES} )
target_compile_features(testing_single_trajectory_decision PRIVATE cxx_range_for)
add_dependencies(testing_single_trajectory_decision ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(testing_single_trajectory_decision ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable( testing_trajectory_decision src/test_src/trajectory_decision.cpp ${MOVEMENT_MANAGER_SRC_FILES} )
target_compile_features(testing_trajectory_decision PRIVATE cxx_range_for)
add_dependencies(testing_trajectory_decision ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(testing_trajectory_decision ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable( testing_vision_processing_image_converter src/test_src/image_converter.cpp )
target_compile_features(testing_vision_processing_image_converter PRIVATE cxx_range_for)
add_dependencies(testing_vision_processing_image_converter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(testing_vision_processing_image_converter ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(testing_vision_receiver src/test_src/video_receiver.cpp )
target_compile_features(testing_vision_receiver PRIVATE cxx_range_for)
target_link_libraries(testing_vision_receiver ${OpenCV_LIBS} ${catkin_LIBRARIES})

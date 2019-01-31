/*
 * manager_single_trajectory.cpp
 *
 *  Created on: 16 Mar 2016
 *      Author: miguel
 */

#include <ros/init.h>
#include <signal.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "../Vision/Color Segmentation/Vision_Processor.h"
#include "Manager_Aux/Movement_Manager.h"

using namespace std;
using namespace movement_decision;

MovementManager* mov_manager;

void signal_handler(int s){

	switch (s) {
		case SIGINT:
			cout << "Caught the " << strsignal(s) << " signal." << endl;
			ros::shutdown();
			delete mov_manager;
			cout << "No problems detected." << endl << "Clean Exit" << endl;
			exit(EXIT_SUCCESS);
			break;
		case SIGSEGV:
			cout << "Caught the " << strsignal(s) << " signal." << endl;
			ros::shutdown();
			delete mov_manager;
			cout << "Error detected during process." << endl;
			exit(EXIT_FAILURE);
		default:
			cout << "Caught the " << strsignal(s) << " signal." << endl;
			break;
	}

}


int main(int argc, char** argv){

	if(argc != 9){
		cout << "[MOVEMENT MANAGER SINGLE TRAJECTORY] usage: ./movement_manager_single_trajectory" <<
				" <cups_pos_subs_topic> <trajectory_type> <new_trajectory_service> <send_traj_service>" <<
				" <robot_feedback_service> <restart_mov_service> <limb_to_move> <mov_period>" << endl;
		return 0;
	}

	string sub_topic = argv[1];
	string trajectory_type = argv[2];
	string trajectory_service = argv[3];
	string start_movement_service = argv[4];
	string robot_feedback_service = argv[5];
	string restart_movement_service = argv[6];
	string robot_limb = argv[7];
	int movement_period = atoi(argv[8]);

	ROS_INFO("Starting Movement Manager: %s", trajectory_type.c_str());

	signal(SIGINT, signal_handler);
	signal(SIGSEGV, signal_handler);

	init(argc, argv, "movement_manager");
	mov_manager = new MovementManager(sub_topic, trajectory_type, trajectory_service, start_movement_service,
			robot_feedback_service, restart_movement_service, robot_limb, movement_period);
	spin();

	return 0;
}


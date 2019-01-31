/*
 * Trajectory_Manager.cpp
 *
 *  Created on: Jun 15, 2016
 *      Author: miguel
 */

#include "Trajectory_Manager.h"

namespace movement_decision {

	Trajectory_Manager::~Trajectory_Manager(){

		_feedback_traj_serv.shutdown();
		_restart_traj_serv.shutdown();
		_send_traj_serv.shutdown();
	}

}  // namespace movement_decision



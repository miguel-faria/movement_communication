/*
 * Single_Trajectory.h
 *
 *  Created on: Jun 15, 2016
 *      Author: miguel
 */

#ifndef SINGLE_TRAJECTORY_H_
#define SINGLE_TRAJECTORY_H_

#include <opencv2/core/core.hpp>
#include <ros/node_handle.h>
#include <ros/service_client.h>
#include <map>
#include <string>

#include "../../../devel/include/robot_serving/Movement.h"
#include "Trajectory_Manager.h"

namespace movement_decision {

	class Single_Trajectory : public Trajectory_Manager {
		private:
			ros::ServiceClient _new_traj_srv;

		public:
			Single_Trajectory() {}

			Single_Trajectory(string trajectory, NodeHandle nh, string traj_srv_name, string send_traj_srv,
					string get_feedback_srv, string restart_srv) :
						Trajectory_Manager(trajectory, nh, send_traj_srv, get_feedback_srv, restart_srv){
				_new_traj_srv = nh.serviceClient<robot_serving::Movement>(traj_srv_name);
			}

			~Single_Trajectory(){
				_feedback_traj_serv.shutdown();
				_new_traj_srv.shutdown();
				_restart_traj_serv.shutdown();
				_send_traj_serv.shutdown();
			}

			ros::ServiceClient get_new_traj_srv( ) const {
				return _new_traj_srv;
			}

			void set_new_traj_srv(ros::ServiceClient new_traj_srv) {
				_new_traj_srv = new_traj_srv;
			}

			ros::ServiceClient choose_trajectory_service(map<string, Point3f> cups, map<string, double> cups_dist,
					string current_cup_id, string arm){
				return _new_traj_srv;
			}
	};

}  // namespace movement_decision



#endif /* SINGLE_TRAJECTORY_H_ */

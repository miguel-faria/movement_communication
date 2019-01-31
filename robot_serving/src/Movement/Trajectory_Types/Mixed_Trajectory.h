/*
 * Mixed_Trajectory.h
 *
 *  Created on: Jun 15, 2016
 *      Author: miguel
 */

#ifndef MIXED_TRAJECTORY_H_
#define MIXED_TRAJECTORY_H_

#include <opencv2/core/core.hpp>
#include <ros/node_handle.h>
#include <ros/service_client.h>
#include <map>
#include <string>

#include "../../../devel/include/robot_serving/Movement.h"
#include "Trajectory_Manager.h"

#define MAX_CUP_DIST 500

namespace movement_decision {


	class Mixed_Trajectory : public Trajectory_Manager {
		private:
			ros::ServiceClient _legible_trajectory_srv;
			ros::ServiceClient _predictable_trajectory_srv;

		public:
			Mixed_Trajectory() {}

			Mixed_Trajectory(string trajectory, NodeHandle nh, string legible_srv_name, string predictable_srv_name,
					string send_traj_srv, string get_feedback_srv, string restart_srv) :
						Trajectory_Manager(trajectory, nh, send_traj_srv, get_feedback_srv, restart_srv){
				_legible_trajectory_srv = nh.serviceClient<robot_serving::Movement>(legible_srv_name);
				_predictable_trajectory_srv = nh.serviceClient<robot_serving::Movement>(predictable_srv_name);
			}

			virtual ~Mixed_Trajectory();

			inline ros::ServiceClient get_legible_trajectory_srv( ) const {
				return _legible_trajectory_srv;
			}

			inline ros::ServiceClient get_predictable_trajectory_srv( ) const {
				return _predictable_trajectory_srv;
			}

			inline void set_legible_trajectory_srv(ros::ServiceClient legible_trajectory_srv) {
				_legible_trajectory_srv = legible_trajectory_srv;
			}

			inline void set_predictable_trajectory_srv(ros::ServiceClient predictable_trajectory_srv) {
				_predictable_trajectory_srv = predictable_trajectory_srv;
			}

			virtual ros::ServiceClient choose_trajectory_service(map<string, Point3f> cups, map<string, double> cups_dist,
					string current_cup_id, string arm);
	};

}  // namespace movement_decision


#endif /* MIXED_TRAJECTORY_H_ */

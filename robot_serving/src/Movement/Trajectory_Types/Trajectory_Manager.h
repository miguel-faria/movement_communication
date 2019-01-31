/*
 * Trajectory_Type.h
 *
 *  Created on: Jun 15, 2016
 *      Author: miguel
 */

#ifndef TRAJECTORY_MANAGER_H_
#define TRAJECTORY_MANAGER_H_

#include <opencv2/core/core.hpp>
#include <ros/node_handle.h>
#include <ros/service_client.h>
#include <map>
#include <string>

#include "../../../devel/include/robot_serving/RobotMovementCancelTrajectory.h"
#include "../../../devel/include/robot_serving/RobotMovementFeedback.h"
#include "../../../devel/include/robot_serving/RobotMovementSendTrajectory.h"

using namespace std;
using namespace ros;
using namespace cv;
using namespace robot_serving;

namespace movement_decision {

	class Trajectory_Manager {
		protected:
			string _trajectory;

			ros::ServiceClient _send_traj_serv;
			ros::ServiceClient _feedback_traj_serv;
			ros::ServiceClient _restart_traj_serv;

		public:
			Trajectory_Manager() {}

			Trajectory_Manager(string trajectory, NodeHandle nh, string send_traj_srv, string get_feedback_srv,
					string restart_srv) : _trajectory(trajectory) {
				_send_traj_serv = nh.serviceClient<robot_serving::RobotMovementSendTrajectory>(send_traj_srv);
				_feedback_traj_serv = nh.serviceClient<robot_serving::RobotMovementFeedback>(get_feedback_srv);
				_restart_traj_serv = nh.serviceClient<robot_serving::RobotMovementCancelTrajectory>(restart_srv);
			}

			virtual ~Trajectory_Manager();

			inline string get_trajectory( ) const {
				return _trajectory;
			}

			inline ros::ServiceClient get_feedback_traj_serv( ) const {
				return _feedback_traj_serv;
			}

			inline ros::ServiceClient get_restart_traj_serv( ) const {
				return _restart_traj_serv;
			}

			inline ros::ServiceClient get_send_traj_serv( ) const {
				return _send_traj_serv;
			}

			inline void set_trajectory(string trajectory) {
				_trajectory = trajectory;
			}

			inline void set_feedback_traj_serv(ros::ServiceClient feedback_traj_serv) {
				_feedback_traj_serv = feedback_traj_serv;
			}

			inline void set_restart_traj_serv(ros::ServiceClient restart_traj_serv) {
				_restart_traj_serv = restart_traj_serv;
			}

			inline void set_send_traj_serv(ros::ServiceClient send_traj_serv) {
				_send_traj_serv = send_traj_serv;
			}

			virtual ros::ServiceClient choose_trajectory_service(map<string, Point3f> cups, map<string, double> cups_dist,
					string current_cup_id, string arm) = 0;

	};

}  // namespace movement_decision



#endif /* TRAJECTORY_MANAGER_H_ */

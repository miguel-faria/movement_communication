/*
 * Mixed_Trajectory.cpp
 *
 *  Created on: Jun 15, 2016
 *      Author: miguel
 */

#include "Mixed_Trajectory.h"

#include <cmath>
#include <cstdlib>

namespace movement_decision {

	Mixed_Trajectory::~Mixed_Trajectory(){
		_feedback_traj_serv.shutdown();
		_legible_trajectory_srv.shutdown();
		_predictable_trajectory_srv.shutdown();
		_restart_traj_serv.shutdown();
		_send_traj_serv.shutdown();
	}

	double euclidean_distance(Point3f ptx1, Point3f ptx2){
		return sqrt(pow(ptx1.x - ptx2.x, 2) + pow(ptx1.y - ptx2.y, 2) + pow(ptx1.z - ptx2.z, 2));
	}

	void get_closer_cups(map<string, Point3f> pos_map, double point_dist, string current_id, Point3f cup, double dist_closer_cup1, double dist_closer_cup2,
			Point3f& cup2, Point3f& cup1) {
		for (map<string, Point3f>::iterator it = pos_map.begin( ); it != pos_map.end( ); it++) {
			point_dist = euclidean_distance(cup, it->second);
			if (point_dist <= dist_closer_cup1 && current_id.compare(it->first) != 0) {
				if (dist_closer_cup1 != INFINITY) {
					dist_closer_cup2 = dist_closer_cup1;
					cup2 = cup1;
				}
				dist_closer_cup1 = point_dist;
				cup1 = it->second;
			}
		}
	}

	bool surrounded(Point3f cup, map<string, Point3f> pos_map, string current_id){

		double dist_closer_cup1 = INFINITY;
		double dist_closer_cup2 = INFINITY;
		double point_dist;
		Point3f cup1, cup2;

		get_closer_cups(pos_map, point_dist, current_id, cup, dist_closer_cup1, dist_closer_cup2, cup2, cup1);

		if (euclidean_distance(cup, cup1) < MAX_CUP_DIST && euclidean_distance(cup, cup2) < MAX_CUP_DIST)
			return true;
		else
			return false;
	}

	bool side_close(Point3f cup, map<string, Point3f> pos_map, string arm, string current_id){

		double dist_closer_cup1 = INFINITY;
		double dist_closer_cup2 = INFINITY;
		double point_dist;
		Point3f cup1, cup2;

		get_closer_cups(pos_map, point_dist, current_id, cup, dist_closer_cup1, dist_closer_cup2, cup2, cup1);

		if(arm.compare("left") == 0 || arm.compare("Left") == 0){
			if ((cup1.x < cup.x && euclidean_distance(cup, cup1) <= MAX_CUP_DIST) || (cup2.x < cup.x && euclidean_distance(cup, cup2) <= MAX_CUP_DIST))
				return true;
		} else {
			if ((cup1.x > cup.x && euclidean_distance(cup, cup1) <= MAX_CUP_DIST) || (cup2.x > cup.x && euclidean_distance(cup, cup2) <= MAX_CUP_DIST))
				return true;
		}

		return false;
	}

	bool outmost_cup(Point3f cup, map<string, Point3f> pos_map, string arm){

		bool left_arm = (arm.compare("left") == 0 || arm.compare("Left") == 0);

		for(map<string, Point3f>::iterator it = pos_map.begin(); it != pos_map.end(); it++){
			if((it->second.x != INFINITY) &&
					((left_arm && it->second.x < cup.x) || (!left_arm && it->second.x > cup.x)))
				return false;
		}

		return true;
	}

	bool equally_spaced_cups(double current_cup_dist, map<string, double> dists, map<string, Point3f> pos){

		for(map<string, Point3f>::iterator it = pos.begin(); it != pos.end(); it++)
			if(dists.at(it->first) != current_cup_dist)
				return false;

		return true;
	}

	ros::ServiceClient Mixed_Trajectory::choose_trajectory_service(map<string, Point3f> cups,
			map<string, double> cups_dist, string current_cup_id, string arm){

		ros::ServiceClient service = ((rand() % 2) == 0 ? _legible_trajectory_srv : _predictable_trajectory_srv);
		map<string, Point3f> pos = map<string, Point3f>(cups);
		map<string, double> dists = map<string, double>(cups_dist);
		Point3f current_cup_pos = pos.find(current_cup_id)->second;
		double current_cup_dist = dists.find(current_cup_id)->second;
		pos.erase(current_cup_id);
		dists.erase(current_cup_id);
		bool surround = surrounded(current_cup_pos, pos, current_cup_id);
		bool laterally_next = side_close(current_cup_pos, pos, arm, current_cup_id);
		bool outmost = outmost_cup(current_cup_pos, pos, arm);
		bool equally_spaced = equally_spaced_cups(current_cup_dist, dists, pos);

		if(pos.size() < 1){
			service = _predictable_trajectory_srv;
		}else if(outmost || (!surround && !laterally_next) || equally_spaced){
			service = _legible_trajectory_srv;
		}else if(surround || laterally_next){
			service = _predictable_trajectory_srv;
		}

		return service;
	}

}  // namespace movement_decision


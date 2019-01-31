/*
 * Movement_Manager.cpp
 *
 *  Created on: May 27, 2016
 *      Author: miguel
 */

#include "Movement_Manager.h"

#include <actionlib_msgs/GoalStatus.h>
#include <opencv2/core/operations.hpp>
#include <ros/service_client.h>
#include <std_srvs/Empty.h>
#include <algorithm>
#include <chrono>
#include <thread>

#include "../../../devel/include/robot_serving/FollowBehaviour.h"
#include "../../../devel/include/robot_serving/Movement.h"
#include "../../../devel/include/robot_serving/RobotMovementCancelTrajectory.h"
#include "../../../devel/include/robot_serving/RobotMovementFeedback.h"
#include "../../../devel/include/robot_serving/RobotMovementSendTrajectory.h"
#include "../Trajectory_Types/Trajectory_Manager.h"

using namespace this_thread;
using namespace chrono;


namespace movement_decision{

	MovementManager::~MovementManager(){
		_sub_cups_pos.shutdown();
		_srvc_timer.stop();
		delete &_traj_manager;
		_log_file.close();

		_cups_dist.clear();
		_cups_observable.clear();
		_cups_pos.clear();
		_cup_mov_phase.clear();
	}

	void MovementManager::update_cup_info(string cup_id, Point3f cup_xyz, double point_dist) {
		_cups_pos.at(cup_id) = cup_xyz;
		_cups_dist.at(cup_id) = point_dist;
		_cups_observable.at(cup_id) = true;
	}

	bool MovementManager::movement_finished(){

		return (_cup_mov_phase.at(_current_cup_id) == Movement_Stage::MOVEMENT_SUCCESS ||
				_cup_mov_phase.at(_current_cup_id) == Movement_Stage::MOVEMENT_FAILED);
	}

	void MovementManager::update_visible_cups(vector<string> cups_id){

		string current_id;

		for(auto it : _cup_time_unseen){

			current_id = it.first;

			if(find(cups_id.begin(), cups_id.end(), current_id) != cups_id.end()){
				it.second = 0;
				if(!_cups_observable.at(current_id))
					_cups_observable.at(current_id) = true;
			}else {
				it.second = min(MAX_TIME_OFF_FRAME, it.second + 1);
				if (it.second == MAX_TIME_OFF_FRAME && _cups_observable.at(current_id))
					_cups_observable.at(current_id) = false;
			}

		}

	}

	void MovementManager::cups_info_receiver(const CupsConstPtr& msg){

		if(msg->cups_pos_x.size() < 1){
			_received_data = false;
			return;
		}else
			_received_data = true;

		int n_cups = msg->n_cups;
		double point_dist;
		vector<float> x_vec, y_vec, z_vec;
		vector<string> cup_ids = msg->cups_color;
		Point3f cup_xyz;
		map<string,Point3f>::iterator pos_it;
		map<string,double>::iterator dist_it;
		map<string,bool>::iterator obs_it;

		x_vec = msg->cups_pos_x;
		y_vec = msg->cups_pos_y;
		z_vec = msg->cups_pos_z;

		ROS_INFO("Cups Info Receiver");

		if(_received_data){
			if(_cups_pos.empty()){

				for(int i = 0; i < n_cups; i++){
					cup_xyz = Point3f(x_vec[i], y_vec[i], z_vec[i]);
					point_dist = sqrt(pow(cup_xyz.x*1000,2) + pow(cup_xyz.y*1000,2) + pow(cup_xyz.z*1000,2));
					insert_cup_info(cup_ids[i], cup_xyz, point_dist);
				}

			} else{
				update_visible_cups(cup_ids);
				for(int i = 0; i < n_cups; i++){
					cup_xyz = Point3f(x_vec[i], y_vec[i], z_vec[i]);
					point_dist = sqrt(pow(cup_xyz.x*1000,2) + pow(cup_xyz.y*1000,2) + pow(cup_xyz.z*1000,2));

					if(_cups_pos.count(cup_ids[i]) == 0){
						insert_cup_info(cup_ids[i], cup_xyz, point_dist);
					} else{
						dist_it = _cups_dist.find(cup_ids[i]);

						//If cup's center of mass moves at least 0.5m update with new info
						if(abs(point_dist - dist_it->second) >= 500){
							update_cup_info(cup_ids[i], cup_xyz, point_dist);
						}
					}
				}
			}
		}
	}

	map<string, Point3f> MovementManager::get_selectable_cups(){

		map<string, Point3f> selectable = map<string, Point3f> (_cups_pos);

		for(map<string, bool>::iterator it = _cups_served.begin(); it != _cups_served.end(); it++)
			selectable.erase(it->first);

		return selectable;
	}

	bool MovementManager::all_cups_served(){
		return _cups_pos.size() == _cups_served.size();
	}

	void MovementManager::movement_decision(const TimerEvent& event){

		if(!_can_start)
			return;
		else
			_start_signal.shutdown();

		map<string, Point3f>::iterator objective_pos_it;
		map<string, Point3f> cups_left = get_selectable_cups();
		int status_response;
		robot_serving::Movement traj_srv;
		RobotMovementSendTrajectory movement_srv;
		RobotMovementFeedback feedback_srv;
		RobotMovementCancelTrajectory restart_srv;
		int movement_result;
		float objective_dist = sqrtf(pow(_current_objective.x, 2) + pow(_current_objective.y, 2)
				+ pow(_current_objective.z, 2));
		ServiceClient cli;
		robot_serving::ManageExpression new_expression;
		robot_serving::SpeechCues new_cue;
		float avg_dist_height;

		ROS_INFO("Movement Decision");
		ROS_INFO("Cups Left: %d", (int)cups_left.size());

		if(_cups_pos.size() > 0){

			//If there is no cup selected or the previous movement has finished
			if((_current_cup_id.empty() || movement_finished()) && !_moving){

				ROS_INFO("Started Decision");

				new_expression.face_code = Facial_Expressions::LOOKING;
				_facial_expression_mng.publish(new_expression);
				ROS_INFO("Changing facial expression to Look for Objects!");

				//Get next cup to serve
				objective_pos_it = choose_next_cup();
				if(objective_pos_it != _cups_pos.end()) {
					_moving = true;
					_current_cup_id = objective_pos_it->first;
					_current_objective = objective_pos_it->second;

					new_expression.face_code = Facial_Expressions::HAPPY;
					_facial_expression_mng.publish(new_expression);
					ROS_INFO("Changing facial expression to Happy and Follow Link!");

					//Call service to move to target
					traj_srv.request.x_pos = _current_objective.x;
					traj_srv.request.y_pos = _current_objective.y + VERTICAL_DISPLACEMENT;
					traj_srv.request.z_pos = _current_objective.z + DEPTH_DISPLACEMENT;

					ROS_INFO("Moving to Target: %s at (%f,%f,%f)", _current_cup_id.c_str(), traj_srv.request.x_pos,
							traj_srv.request.y_pos, traj_srv.request.z_pos);

					_log_file << "MOVEMENT MANAGER: MOVEMENT DECISION" << endl;
					_log_file << "Target " << _current_cup_id << ": (" << traj_srv.request.x_pos << ", " <<
							traj_srv.request.y_pos << ", " << traj_srv.request.z_pos << ") \tDist: " <<
							_cups_dist.at(_current_cup_id) << endl;

					cli = _traj_manager->choose_trajectory_service(cups_left, _cups_dist,
							_current_cup_id, _baxter_limb);

					if(cli.call(traj_srv)){
						movement_result = traj_srv.response.movement_success;
						ROS_INFO("Response from movement trajectory service received.\n");
					}else {
						ROS_ERROR("Failed to call service to obtain new trajectory.\n");
						_current_cup_id = string();
						_current_objective = Point3f();
					}

					_log_file << "Result " << _current_cup_id << ": " << movement_result << endl;
					_cups_served.insert(std::pair<string, bool>(_current_cup_id, true));
					_log_file << "Added cup: " << _current_cup_id << " to successfull list!" << endl;
					_log_file << "Served " << (int)_cups_served.size() << " cups." << endl;

					if(movement_result == Movement_Result::SUCCESS){
						_cup_mov_phase.at(_current_cup_id) = Movement_Stage::MOVEMENT_SUCCESS;
						_current_cup_id = string();
						_current_objective = Point3f();
						//new_expression.face_code = Facial_Expressions::SUCCEEDED;
						ROS_INFO("Changing facial expression to Very Happy!");
					}else if(movement_result == Movement_Result::ABORTED || movement_result == Movement_Result::PREEMPTED){
						_cup_mov_phase.at(_current_cup_id) = Movement_Stage::MOVEMENT_FAILED;
						_current_cup_id = string();
						_current_objective = Point3f();
						//new_expression.face_code = Facial_Expressions::SAD;
						ROS_INFO("Changing facial expression to Sad!");
					}
					//_facial_expression_mng.publish(new_expression);
					if(_counter_served < 1){
						new_cue.speech_cue_code = Speech_Interactions::NEXT;
					}else if(((int)cups_left.size() - 1) < 1){
						new_cue.speech_cue_code = Speech_Interactions::LAST_CUP;
					}else {
						new_cue.speech_cue_code = Speech_Interactions::NEXT2;
					}
					_speech_cues_mng.publish(new_cue);
					_counter_served++;
					std::this_thread::sleep_for(std::chrono::milliseconds(1500));
					_moving = false;
				}else{
					_moving = true;
					if(!_received_data){
						new_cue.speech_cue_code = Speech_Interactions::SMALL_TALK;
						_speech_cues_mng.publish(new_cue);
						_log_file << "Small talk." << endl;
						ROS_INFO("No Cup in Robot's range and no visual data received.");
					}else {
						if(((int)_cups_pos.size() - (int)_cups_served.size()) > 1){
							avg_dist_height = 0;
							for(map<string, Point3f>::iterator it = cups_left.begin(); it != cups_left.end(); it++)
								avg_dist_height += sqrt(pow(it->second.y, 2) + pow(it->second.z, 2));
							avg_dist_height = (avg_dist_height / (float)((int)cups_left.size()));
							if(avg_dist_height > MAX_VERTICAL_HEIGTH || _closer_counter > 2){
								new_cue.speech_cue_code = Speech_Interactions::RAISE_CUP_MULTI;
								_log_file << "Asked users to raise the cups." << endl;
							} else{
								new_cue.speech_cue_code = Speech_Interactions::COME_CLOSER_MULTI;
								_closer_counter++;
								_log_file << "Asked users to come closer." << endl;
							}
						}else {
							if(sqrt(pow(cups_left.begin()->second.y, 2) + pow(cups_left.begin()->second.z, 2)) > MAX_VERTICAL_HEIGTH){
								new_cue.speech_cue_code = Speech_Interactions::RAISE_CUP;
								_log_file << "Asked last user to raise the cup." << endl;
							}else if(_closer_counter > 2){
								if(!_asked_straight){
									new_cue.speech_cue_code = Speech_Interactions::STRAIGHTEN_CUP;
									_asked_straight = true;
									_log_file << "Asked last user to straighten the cup." << endl;
								} else {
									new_cue.speech_cue_code = Speech_Interactions::RAISE_CUP;
									_log_file << "Asked last user to raise the cup." << endl;
								}
							} else {
								new_cue.speech_cue_code = Speech_Interactions::COME_CLOSER;
								_closer_counter++;
								_log_file << "Asked last user to come closer." << endl;
							}
						}
						_speech_cues_mng.publish(new_cue);
						ROS_INFO("No Cup in Robot's range");
						_moving = false;
					}
				}


			//If there's already a selected target
			} else{

				//Check movement state
				feedback_srv.request.limb = _baxter_limb;
				if(_traj_manager->get_feedback_traj_serv().call(feedback_srv)){
					status_response = feedback_srv.response.feedback;
					ROS_INFO("Response from the get feedback service about the %s limb received.\n",
							_baxter_limb.c_str());
				}else{
					ROS_ERROR("Failed to call service to get feed back on movement by the %s limb.\n",
							_baxter_limb.c_str());
				}

				if(status_response == actionlib_msgs::GoalStatus::SUCCEEDED){
					_cup_mov_phase.at(_current_cup_id) = Movement_Stage::MOVEMENT_SUCCESS;
					_log_file << "Goal Status: Movement was a success." << endl;
					return;
				} else if(status_response == actionlib_msgs::GoalStatus::ABORTED){
					_cup_mov_phase.at(_current_cup_id) = Movement_Stage::MOVEMENT_FAILED;
					_log_file << "Goal Status: Movement was a failure." << endl;
					return;
				} else if(status_response == actionlib_msgs::GoalStatus::ACTIVE){

					_log_file << "Goal Status: Goal is still active." << endl;

					//Verify if there were changes in the cup's position and visibility
					if(abs(objective_dist - _cups_dist.at(_current_cup_id)) >= 500 &&
							_cups_observable.at(_current_cup_id)){

						new_expression.face_code = Facial_Expressions::LOOKING;
						_facial_expression_mng.publish(new_expression);

						//If position changes significantly calculate trajctory to new target
						//Cancel current movement
						restart_srv.request.limb = _baxter_limb;
						restart_srv.request.service_code = CANCEL_MOVEMENT;
						if(_traj_manager->get_restart_traj_serv().call(restart_srv)){
							_moving = true;
							ROS_INFO("%f\t%f", VERTICAL_DISPLACEMENT, DEPTH_DISPLACEMENT);

							//Find new target and start movement
							_current_objective = _cups_pos.at(_current_cup_id);
							traj_srv.request.x_pos = _current_objective.x;
							traj_srv.request.y_pos = _current_objective.y + VERTICAL_DISPLACEMENT;
							traj_srv.request.z_pos = _current_objective.z + DEPTH_DISPLACEMENT;

							new_expression.face_code = Facial_Expressions::HAPPY;
							_facial_expression_mng.publish(new_expression);

							ROS_INFO("Moving to Target: %s at (%f,%f,%f)", _current_cup_id.c_str(), traj_srv.request.x_pos,
									traj_srv.request.y_pos, traj_srv.request.z_pos);

							_log_file << "MOVEMENT MANAGER: MOVEMENT DECISION" << endl;
							_log_file << "Target " << _current_cup_id << ": (" << traj_srv.request.x_pos << ", " <<
									traj_srv.request.y_pos << ", " << traj_srv.request.z_pos << ") \tDist: " <<
									_cups_dist.at(_current_cup_id) << endl;

							cli = _traj_manager->choose_trajectory_service(_cups_pos, _cups_dist,
									_current_cup_id, _baxter_limb);


							if(cli.call(traj_srv)){
								movement_result = traj_srv.response.movement_success;
								ROS_INFO("Response from movement trajectory service received.\n");
							}else {
								ROS_ERROR("Failed to call service to obtain new trajectory.\n");
								_current_cup_id = string();
								_current_objective = Point3f();
							}

							_log_file << "Result " << _current_cup_id << ": " << movement_result << endl;

							if(movement_result == Movement_Result::SUCCESS){
								_cup_mov_phase.at(_current_cup_id) = Movement_Stage::MOVEMENT_SUCCESS;
								_current_cup_id = string();
								_current_objective = Point3f();
								new_expression.face_code = Facial_Expressions::SUCCEEDED;
							}else if(movement_result == Movement_Result::ABORTED || movement_result == Movement_Result::PREEMPTED){
								_cup_mov_phase.at(_current_cup_id) = Movement_Stage::MOVEMENT_FAILED;
								_current_cup_id = string();
								_current_objective = Point3f();
								new_expression.face_code = Facial_Expressions::SAD;
							}
							_facial_expression_mng.publish(new_expression);
							_moving = false;
						}else{
							ROS_ERROR("Failed to call service to cancel movement.\n");
						}
					}
				}
			}

			if(all_cups_served()){
				ros::ServiceClient fis = _nh.serviceClient<std_srvs::Empty>("/finish_interaction_server");
				ros::ServiceClient fms = _nh.serviceClient<std_srvs::Empty>("/finish_movement_server");
				std_srvs::Empty ept_interaction;
				std_srvs::Empty ept_move;
				bool interaction_server_off = false;
				bool movement_server_off = false;
				int timeout_counter = 0;
				while(timeout_counter < 10 && (!interaction_server_off || !movement_server_off)){
					if(!interaction_server_off){
						if(fis.call(ept_interaction)){
							interaction_server_off = true;
							ROS_INFO("Call for finishing interaction server execution success!!");
						}else{
							ROS_INFO("Failed to call to finish interaction server execution!!");
						}
					}
					if(!movement_server_off){
						if(fms.call(ept_move)){
							movement_server_off = true;
							ROS_INFO("Call for finishing movement server execution success!!");
						}else{
							ROS_INFO("Failed to call to finish movement server execution!!");
						}
					}
					timeout_counter++;
				}

				exit(0);
			}
		}

	}

	void MovementManager::follow_movement_interaction(const TimerEvent& event){

		ros::Publisher	follow_mode_pub = _nh.advertise<robot_serving::FollowBehaviour>("bea/follow_mode", 1000);
		robot_serving::FollowBehaviour fb;

		if(_current_cup_id.empty() || movement_finished()){

			fb.name = "Looking for Objects";
			fb.follow_rule = FollowBehaviour::FOLLOW_RANDOM_LOOK;

		}else{

			fb.name = "Following Movement";
			fb.follow_rule = FollowBehaviour::FOLLOW_LINK;

		}

		fb.with_head = false;
		fb.with_eyes = true;

		follow_mode_pub.publish(fb);

	}
}

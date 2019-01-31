/*
 * Movement_Manager.h
 *
 *  Created on: May 20, 2016
 *      Author: miguel
 */

#ifndef MOVEMENT_DECISION_MOVEMENT_MANAGER_H_
#define MOVEMENT_DECISION_MOVEMENT_MANAGER_H_

#include <opencv2/core/core.hpp>
#include <ros/duration.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <ros/timer.h>
#include <rosconsole/macros_generated.h>
#include <std_msgs/Int32.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../../devel/include/robot_serving/Cups.h"
#include "../../../devel/include/robot_serving/ManageExpression.h"
#include "../../../devel/include/robot_serving/PMPTraj.h"
#include "../../../devel/include/robot_serving/SpeechCues.h"
#include "../Trajectory_Types/Mixed_Trajectory.h"
#include "../Trajectory_Types/Single_Trajectory.h"

using namespace std;
using namespace ros;
using namespace cv;
using namespace robot_serving;

namespace movement_decision {

#define CANCEL_MOVEMENT 		1
#define RESTART_MOVEMENT 		2
#define MAX_TIME_OFF_FRAME  	5
#define MAX_REACH				1500
#define VERTICAL_DISPLACEMENT	0.0
#define DEPTH_DISPLACEMENT		-0.035
#define MAX_VERTICAL_HEIGTH		1.3

	class MovementManager {

		private:

			//Possible stages of the movement of the robot towards a cup
			enum Movement_Stage {
				QUEUED,
				STARTED_MOVEMENT,
				MOVEMENT_SUCCESS,
				MOVEMENT_FAILED
			};

			enum Movement_Result {
				ABORTED,
				SUCCESS,
				PREEMPTED
			};

			enum Facial_Expressions{
				HAPPY = 1,
				SAD,
				NEUTRAL,
				SUCCEEDED,
				LOOKING
			};

			enum Speech_Interactions{
				RAISE_CUP = 1,
				COME_CLOSER,
				STRAIGHTEN_CUP,
				RAISE_CUP_MULTI,
				COME_CLOSER_MULTI,
				NEXT,
				NEXT2,
				LAST_CUP,
				SMALL_TALK
			};

			bool _received_data = false;
			bool _can_start = false;
			bool _asked_straight = false;
			bool _moving = false;

			int _closer_counter = 0;
			int _counter_served = 0;

			ros::NodeHandle _nh;
			ros::Subscriber _sub_cups_pos;
			ros::Subscriber _start_signal;
			ros::Publisher _facial_expression_mng;
			ros::Publisher _speech_cues_mng;
			ros::Timer _srvc_timer;
			ros::Timer _baxter_social_reaction;
			Trajectory_Manager *_traj_manager;

			map<string, Point3f> _cups_pos;
			map<string, double> _cups_dist;
			map<string, bool> _cups_observable;
			map<string, bool> _cups_served;
			map<string, int> _cup_mov_phase;
			map<string, int> _cup_time_unseen;

			PMPTraj _robot_movement;
			Point3f _current_objective;
			string _current_cup_id;
			string _baxter_limb;
			ofstream _log_file;


			virtual void update_visible_cups(vector<string> cups_id);
			virtual bool all_cups_served();

			/*
			 * update_cup_info - private method used to update the xyz position, distance and visibility status of
			 * a specific cup
			 *
			 * params:
			 * 		- cup_id 	 - string with the identification of the cup
			 * 		- cup_xyz 	 - 3D position of the cup
			 * 		- point_dist - Euclidean distance between the robot and the cup position
			 */
			virtual void update_cup_info(string cup_id, Point3f cup_xyz, double point_dist);

			/*
			 * movement_finished - private method that verifies if the current robot movement towards the target has
			 * finished, either in success or in failure
			 *
			 * return:
			 * 		- true	- if the movement has finished
			 * 		- false	- if the movement is still under way or waiting to be started
			 */
			virtual bool movement_finished();

		public:

			MovementManager() {};

			MovementManager(string cups_sub_topic, string traj_type, string traj_srvc_name, string send_traj_srv,
					string get_feedback_srv, string restart_srv, string baxter_limb, int timer_time) {

				_sub_cups_pos = _nh.subscribe(cups_sub_topic, 1000, &MovementManager::cups_info_receiver, this);
				_start_signal = _nh.subscribe("/start", 100, &MovementManager::start_handler, this);

				_facial_expression_mng = _nh.advertise<robot_serving::ManageExpression>("facial_expression_mng", 1000);
				_speech_cues_mng = _nh.advertise<robot_serving::SpeechCues>("speech_cues_mng", 1000);

				_traj_manager = new Single_Trajectory(traj_type, _nh, traj_srvc_name,
						send_traj_srv, get_feedback_srv, restart_srv);

				_srvc_timer = _nh.createTimer(ros::Duration(timer_time), &MovementManager::movement_decision, this);
				_baxter_social_reaction = _nh.createTimer(ros::Duration(timer_time), &MovementManager::follow_movement_interaction, this);

				_cups_pos = map<string, Point3f>();
				_cups_dist = map<string, double>();
				_cups_observable = map<string, bool>();
				_cups_served = map<string, bool>();
				_cup_mov_phase = map<string, int>();
				_baxter_limb = baxter_limb;
				_current_cup_id = string();
				_log_file.open("movement_log.txt");
			}

			MovementManager(string cups_sub_topic, string traj_type, string legible_traj_srv,
					string predictable_traj_srv, string send_traj_srv, string get_feedback_srv, string restart_srv,
					string baxter_limb, int timer_time) {

				_sub_cups_pos = _nh.subscribe(cups_sub_topic, 1000, &MovementManager::cups_info_receiver, this);
				_start_signal = _nh.subscribe("/start", 100, &MovementManager::start_handler, this);

				_facial_expression_mng = _nh.advertise<robot_serving::ManageExpression>("facial_expression_mng", 1000);
				_speech_cues_mng = _nh.advertise<robot_serving::SpeechCues>("speech_cues_mng", 1000);

				_traj_manager = new Mixed_Trajectory(traj_type, _nh, legible_traj_srv, predictable_traj_srv, send_traj_srv,
						get_feedback_srv, restart_srv);

				_srvc_timer = _nh.createTimer(ros::Duration(timer_time), &MovementManager::movement_decision, this);
				_baxter_social_reaction = _nh.createTimer(ros::Duration(timer_time), &MovementManager::follow_movement_interaction, this);

				_cups_pos = map<string, Point3f>();
				_cups_dist = map<string, double>();
				_cups_observable = map<string, bool>();
				_cups_served = map<string, bool>();
				_cup_mov_phase = map<string, int>();
				_baxter_limb = baxter_limb;
				_current_cup_id = string();
				_log_file.open("movement_log.txt");
			}

			virtual ~MovementManager();

			void insert_cup_info(string cup_id, Point3f xyz, double dist) {
				_cups_pos.insert(pair<string, Point3f>(cup_id, xyz));
				_cups_dist.insert(pair<string, double>(cup_id, dist));
				_cups_observable.insert(pair<string, bool>(cup_id, true));
				_cup_time_unseen.insert(pair<string, int>(cup_id, 0));
				_cup_mov_phase.insert(pair<string, int>(cup_id, Movement_Stage::QUEUED));
			}

			template<class T, class U> static bool compare_map_elems(pair<T, U> i, pair<T, U> j){
					return i.second < j.second;
			}

			bool infinite(string cup_id){

				Point3f cup_pos = _cups_pos.at(cup_id);

				return (cup_pos.x == INFINITY || cup_pos.y == INFINITY || cup_pos.z == INFINITY) ? true : false;
			}

			bool reacheable(string cup_id){
				return _cups_dist.at(cup_id) <= MAX_REACH ? true : false;
			}

			map<string, Point3f>::iterator choose_next_cup(){
				//map<string, Point3f> cups = map<string, Point3f>(_cups_pos);
				map<string, Point3f> cups = get_selectable_cups();
				string select_cup_id;
				bool selected = false;
				map<string, Point3f>::iterator it;
				srand(time(0));
				while(!selected && cups.size() > 0){
					it = cups.begin();
					std::advance(it, rand() % cups.size());
					if(_cups_served.find(it->first)->second)
						cups.erase(it->first);
					else{
						ROS_INFO("Cup %s is at %f", it->first.c_str(), _cups_dist.at(it->first));
						if((_cup_mov_phase.find(it->first)->second == Movement_Stage::QUEUED) &&
								(_cups_observable.find(it->first)->second) && !infinite(it->first)
								&& reacheable(it->first)){
							select_cup_id = it->first;
							selected = true;
						}else{
							cups.erase(it->first);
						}
					}
				}
				if(selected){
					return _cups_pos.find(select_cup_id);
				}else {
					return _cups_pos.end();
				}
			}


			virtual void cups_info_receiver(const robot_serving::CupsConstPtr& msg);
			virtual void movement_decision(const TimerEvent& event);
			virtual void follow_movement_interaction(const TimerEvent& event);

			virtual map<string, Point3f> get_selectable_cups();

			void start_handler(const std_msgs::Int32::ConstPtr& msg){
				ROS_INFO("Start Handler!!");
				if(msg->data == 1)
					_can_start = true;
			}

	};


}  // namespace movement_decision



#endif /* MOVEMENT_MANAGER_H_ */

/*
 * vision_processing.cpp
 *
 *  Created on: 2 Mar 2016
 *      Author: miguel
 */

#include <ros/duration.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/rate.h>
#include <ros/this_node.h>
#include <ros/timer.h>
#include <signal.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../../../include/Color_Segmentation.h"
#include "Image_Converter.h"
#include "Vision_Processor.h"

using namespace vision_processing_color_seg;

bool clean_quit = false; //signal clean exit flag
bool error_quit = false; //signal error exit flag
Vision_Processor *vp;


bool is_quit(){
	return (clean_quit || error_quit);
}

void signal_handler(int s){

	switch (s) {
		case SIGINT:
			cout << "Caught the " << strsignal(s) << " signal." << endl;
			ros::shutdown();
			delete vp;
			cout << "No problems detected." << endl << "Clean Exit" << endl;
			exit(EXIT_SUCCESS);
			break;
		case SIGSEGV:
			cout << "Caught the " << strsignal(s) << " signal." << endl;
			ros::shutdown();
			delete vp;
			cout << "Error detected during process." << endl;
			exit(EXIT_FAILURE);
		default:
			cout << "Caught the " << strsignal(s) << " signal." << endl;
			break;
	}

}

int main(int argc, char **argv){

	if ( argc != 5 ){
		cout << "[VISION PROCESSING] usage: ./vision_processing <subscribe_color_image_topic_name>" <<
				" <subscribe_depth_image_topic_name> <publish_topic_name> <camera_calibration_params_file_path>"
				<< endl;
		return -1;
	}

	string subs_color_topic_name = argv[1];
	string subs_depth_topic_name = argv[2];
	string calibration_params_file = argv[4];
	string pub_topic_name;
	Red_HSV_Color_Segmentation* red_segmentation = new Red_HSV_Color_Segmentation(vector<int>{RED_LEFT_LOW_H, RED_LEFT_HIGH_H},
			vector<int>{RED_RIGHT_LOW_H, RED_RIGHT_HIGH_H},
			vector<int>{RED_LOW_S, RED_HIGH_S},
			vector<int>{RED_LOW_V, RED_HIGH_V});
	HSV_Color_Segmentation* green_segmentation = new HSV_Color_Segmentation(vector<int>{GREEN_LOW_H, GREEN_HIGH_H},
			vector<int>{GREEN_LOW_S, GREEN_HIGH_S},
			vector<int>{GREEN_LOW_V, GREEN_HIGH_V},
			"green");
	HSV_Color_Segmentation* blue_segmentation = new HSV_Color_Segmentation(vector<int>{BLUE_LOW_H, BLUE_HIGH_H},
			vector<int>{BLUE_LOW_S, BLUE_HIGH_S},
			vector<int>{BLUE_LOW_V, BLUE_HIGH_V},
			"blue");

	//ROS node initialization
	init(argc, argv, "vision_processing");

	//Setting up Vision Processing unit
	vp = new Vision_Processor(calibration_params_file, subs_color_topic_name,
			subs_depth_topic_name);											//Connecting to subscribe topic
	signal(SIGINT, signal_handler);											//Create handler for termination signal
	signal(SIGSEGV, signal_handler);										//Create handler for segmentation fault
	pub_topic_name = this_node::getName() + "/" + argv[3];
	vp->set_cups_publish_topic(pub_topic_name);								//Starting advertising in publish topic
	vp->add_segmentation(red_segmentation);									//Configurating image segmentation
	vp->add_segmentation(green_segmentation);								//configurations
	vp->add_segmentation(blue_segmentation);
	vp->set_timer(vp->get_ic().get_nh().createTimer(ros::Duration(2.5),
			&Vision_Processor::process_and_send, vp));						//Setting timed processing of frames

	Rate loopRate(15);
	while(ok() && !is_quit()){
		spinOnce();
		loopRate.sleep();
	}

	if(clean_quit){
		cout << "No problems detected." << endl << "Clean Exit" << endl;
	} else if(error_quit){
		cout << "Error detected during process." << endl;
	}else {
		cout << "Exit due to problem in ROS environment" << endl;
	}

	vp->stop_timer();
	delete vp;
	ros::shutdown();

	return 0;
}

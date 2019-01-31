/*
 * cup_identification_recording.cpp
 *
 *  Created on: Jun 28, 2016
 *      Author: miguel
 */

#include <Color_Segmentation.h>
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

#include "../src/Vision/Color Segmentation/Image_Converter.h"
#include "../src/Vision/Color Segmentation/Vision_Processor.h"
#include "../src/Vision/Color Segmentation/Color_Constants.h"

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
	HSV_Color_Segmentation* cup2_segmentation = new HSV_Color_Segmentation(vector<int>{BLUE_LOW_HUE,BLUE_HIGH_HUE},
				vector<int>{BLUE_LOW_SAT,BLUE_HIGH_SAT},
				vector<int>{BLUE_LOW_VAL,BLUE_HIGH_VAL},
				"cup2 - Blue");

	//ROS node initialization
	init(argc, argv, "vision_processing");

	//Setting up Vision Processing unit
	vp = new Vision_Processor(calibration_params_file, subs_color_topic_name,
			subs_depth_topic_name);											//Connecting to subscribe topic
	signal(SIGINT, signal_handler);											//Create handler for termination signal
	signal(SIGSEGV, signal_handler);										//Create handler for segmentation fault
	pub_topic_name = this_node::getName() + "/" + argv[3];
	vp->set_cups_publish_topic(pub_topic_name);								//Starting advertising in publish topic
	vp->add_segmentation(cup2_segmentation);
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

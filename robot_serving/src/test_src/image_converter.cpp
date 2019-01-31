/*
 * image_converter.cpp
 *
 *  Created on: 11 Mar 2016
 *      Author: miguel
 */

#include <ros/init.h>
#include <iostream>

#include "../src/Vision/Color Segmentation/Image_Converter.h"

using namespace vision_processing_color_seg;

int main(int argc, char** argv){

	if(argc != 2) {
		cout << "usage: ./image_converter <subscribe_ros_topic_name>" << endl;
		return -1;
	}

	ros::init(argc, argv, "image_converter");
	ImageConverter ic(argv[1]);
	ros::spin();
	return 0;
}



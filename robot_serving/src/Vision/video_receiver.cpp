/*
 * vision_receiver.cpp
 *
 *  Created on: 2 Mar 2016
 *      Author: miguel
 */

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/publisher.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/rate.h>
#include <ros/this_node.h>
#include <ros/time.h>
#include <rosconsole/macros_generated.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace ros;
using namespace std_msgs;
using namespace image_transport;

class Video_Sender{

private:
	NodeHandle _node;
	ImageTransport _im_trans;
	image_transport::Publisher _pub;

public:

	Video_Sender(string pub_topic) : _im_trans(_node) {
		_pub = _im_trans.advertise(pub_topic, 1000);
	}

	inline ImageTransport getImTrans() const {
		return _im_trans;
	}

	inline NodeHandle getNode() const {
		return _node;
	}

	inline image_transport::Publisher getPub() const {
		return _pub;
	}

	inline void setImTrans(ImageTransport imTrans) {
		_im_trans = imTrans;
	}

	inline void setNode(NodeHandle node) {
		_node = node;
	}

	inline void setPub(image_transport::Publisher pub) {
		_pub = pub;
	}

	int send_video_frame(VideoCapture video_src){
		Mat frame;
		sensor_msgs::ImagePtr out_msg;

		if(_pub.getTopic() == ""){
			cout << "No topic associated!!" << endl;
			return -1;
		}

		bool capture_success = video_src.read(frame);
		if(!capture_success){
			cout << "Frame capture failed, skipping to next one!" << endl;
			return -1;
		}

		out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
		_pub.publish(out_msg);

		return 0;
	}
};

int main(int argc, char **argv){

	if ( argc != 4 ){
		cout << "usage: ./vision_receiver -f <video_source> <ros_topic_name>  or"
				<< " ./vision_receiver -d <recording_camera> <ros_topic_name>" << endl;
		return -1;
	}

	VideoCapture cap;
	string opt = argv[1];
	string topic_name;

	if (opt.compare("-f") == 0) {
		cap.open(argv[2]);
	} else if (opt.compare("-d") == 0) {
		cap.open(atoi(argv[2]));
	} else {
		cout << "Invalid option! Please use -f for a file and -d for a recording device." << endl;
		return -1;
	}

	if(!cap.isOpened()){
		cout << "Could not open Video Capture" << endl;
		return -1;
	}

	//Initialize Ros
	init(argc, argv, "video_sender");
	topic_name = ros::this_node::getName() + "/" + argv[3];
	Video_Sender vs(topic_name);
	Rate loop_rate(25);
	int error_counter = 0;
	while(ok() && error_counter <= 10){
		if(vs.send_video_frame(cap) != 0){
			ROS_INFO("Error sending new video frame!!");
			error_counter++;
		} else{
			ROS_INFO("Video frame sent!");
			if(error_counter != 0)
				error_counter = 0;
		}
		spinOnce();
		loop_rate.sleep();
	}

	return 0;
}

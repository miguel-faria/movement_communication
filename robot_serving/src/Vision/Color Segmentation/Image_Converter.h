/*
 * Image_Converter.h
 *
 *  Created on: 12 Mar 2016
 *      Author: miguel
 */

#ifndef VISION_PROCESSING_COLOR_SEG_IMAGE_CONVERTER_H_
#define VISION_PROCESSING_COLOR_SEG_IMAGE_CONVERTER_H_

#include <boost/smart_ptr/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <rosconsole/macros_generated.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

static const string COLOR_FRAME_WINDOW = "Received Color Image";
static const string DEPTH_FRAME_WINDOW = "Received Depth Image";

namespace vision_processing_color_seg {

	class ImageConverter {
		private:
			ros::NodeHandle _nh;

			//Color Subscriber
			image_transport::ImageTransport _it_color;
			image_transport::Subscriber _image_sub_color;
			Mat _color_frame;

			//Depth Subscriber
			image_transport::ImageTransport _it_depth;
			image_transport::Subscriber _image_sub_depth;
			Mat _depth_frame;
			sensor_msgs::Image::_encoding_type _depth_encoding;

			//Publisher
			ros::Publisher _out_pub;

			float _ratio_height;
			float _ratio_width;
			bool _received_depth = false;


		public:

			ImageConverter() : _it_color(_nh), _it_depth(_nh), _ratio_height(0), _ratio_width(0) {
				namedWindow(COLOR_FRAME_WINDOW, CV_WINDOW_NORMAL);
			}

			ImageConverter(string sub_color_topic) : _it_color(_nh), _it_depth(_nh), _ratio_height(0), _ratio_width(0) {
				// Subscribe to input video feed
				_image_sub_color = _it_color.subscribe(sub_color_topic, 1000, &ImageConverter::color_frame_receiver, this);

				namedWindow(COLOR_FRAME_WINDOW, CV_WINDOW_NORMAL);
			}

			ImageConverter(string sub_color_topic, string sub_depth_topic) : _it_color(_nh), _it_depth(_nh),
					_ratio_height(0), _ratio_width(0) {
				// Subscribe to input video feed
				_image_sub_color = _it_color.subscribe(sub_color_topic, 1000, &ImageConverter::color_frame_receiver, this);
				_image_sub_depth = _it_depth.subscribe(sub_depth_topic, 1000, &ImageConverter::depth_frame_receiver, this);

				namedWindow(COLOR_FRAME_WINDOW, CV_WINDOW_NORMAL);
			}

			~ImageConverter()	{
				destroyWindow(COLOR_FRAME_WINDOW);
			}

			void color_frame_receiver(const sensor_msgs::ImageConstPtr& msg)	{
				cv_bridge::CvImagePtr cv_ptr;
				try {
					cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
				} catch (cv_bridge::Exception& e) {
					ROS_ERROR("cv_bridge exception: %s", e.what());
					return;
				}

				try{
					_color_frame = cv_ptr->image;
					cv::imshow(COLOR_FRAME_WINDOW, _color_frame);
					resizeWindow(COLOR_FRAME_WINDOW, 960, 540);
					cv::waitKey(3);
				} catch (ros::Exception& e) {
					ROS_ERROR("ros exception at color_frame_receiver: %s", e.what());
					return;
				}
			}

			void print_mat(Mat image){
				Point2i ptr;
				for (int i = 0; i < image.rows; ++i) {
					cout << "[ ";
					for (int j = 0; j < image.cols; ++j) {
						ptr = Point2i(i,j);
						cout << image.at<int>(ptr) << " ";
					}
					cout << "]" << endl;
				}
			}

			void depth_frame_receiver(const sensor_msgs::ImageConstPtr& msg)	{
				cv_bridge::CvImagePtr cv_ptr;
				try {
					cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
				} catch (cv_bridge::Exception& e) {
					ROS_ERROR("cv_bridge exception: %s", e.what());
					return;
				}


				try{
					cv::imshow(DEPTH_FRAME_WINDOW, cv_ptr->image);
					cv::waitKey(3);

					_depth_frame = cv_ptr->image;
					_depth_encoding = msg->encoding;

					_ratio_height = float(_color_frame.rows)/float(_depth_frame.rows);
					_ratio_width = float(_color_frame.cols)/float(_depth_frame.cols);
					_received_depth = true;

				} catch (ros::Exception& e) {
					ROS_ERROR("ros exception at depth_frame_receiver: %s", e.what());
					return;
				}
			}

			inline ros::NodeHandle get_nh() const {
				return _nh;
			}

			inline Mat get_color_frame() const {
				return _color_frame;
			}

			inline ros::Publisher get_out_pub() const {
				return _out_pub;
			}

			inline image_transport::Subscriber get_image_color_sub() const {
				return _image_sub_color;
			}

			inline Mat get_depth_frame() const {
				return _depth_frame;
			}

			inline image_transport::Subscriber get_image_sub_depth() const {
				return _image_sub_depth;
			}

			inline image_transport::ImageTransport get_it_depth() const {
				return _it_depth;
			}

			inline float get_ratio_height( ) const {
				return _ratio_height;
			}

			inline float get_ratio_width( ) const {
				return _ratio_width;
			}

			inline void set_color_frame(Mat frame) {
				_color_frame = frame;
			}

			inline void set_out_pub(ros::Publisher out_pub) {
				_out_pub = out_pub;
			}

			inline void set_image_color_sub(image_transport::Subscriber imageSub) {
				_image_sub_color = imageSub;
			}

			void assign_sub_color_topic(string sub_topic){
				_image_sub_color = _it_color.subscribe(sub_topic, 1000, &ImageConverter::color_frame_receiver, this);
			}

			void assign_sub_depth_topic(string sub_topic){
				_image_sub_depth = _it_depth.subscribe(sub_topic, 1000, &ImageConverter::depth_frame_receiver, this);
			}

			void set_depth_frame(Mat depth_frame) {
				_depth_frame = depth_frame;
			}

			void set_image_sub_depth(image_transport::Subscriber image_sub_depth) {
				_image_sub_depth = image_sub_depth;
			}

			void set_it_depth(image_transport::ImageTransport it_depth) {
				_it_depth = it_depth;
			}

			void set_ratio_height(float ratio_height) {
				_ratio_height = ratio_height;
			}

			void set_ratio_width(float ratio_width) {
				_ratio_width = ratio_width;
			}

			bool is_received_depth( ) const {
				return _received_depth;
			}

			void set_received_depth(bool received_depth = false) {
				_received_depth = received_depth;
			}

			template <typename T> void assign_pub_topic(string out_topic){
				_out_pub = _nh.advertise<T>(out_topic, 1000, true);
			}
	};

}  // namespace vision_processing_color_seg

#endif /* SRC_VISION_COLOR_SEGMENTATION_IMAGE_CONVERTER_H_ */

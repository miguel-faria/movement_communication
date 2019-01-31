/*
 * Color_Segmentation_Vision_Processor.h
 *
 *  Created on: Apr 20, 2016
 *      Author: miguel
 */

#ifndef VISION_PROCESSING_COLOR_SEG_VISION_PROCESSOR_H_
#define VISION_PROCESSING_COLOR_SEG_VISION_PROCESSOR_H_

#include <Color_Segmentation.h>
#include <opencv2/core/core.hpp>
#include <ros/timer.h>
#include <cmath>
#include <string>
#include <vector>

#include "../../../devel/include/robot_serving/Cups.h"
#include "Camera_Parameters.h"
#include "Image_Converter.h"


using namespace ros;
using namespace color_seg;
using namespace robot_serving;

#define MAX_DEPTH 						8000
#define BASE_DEPTH_WIDTH_RESOLUTION 	512
#define BASE_DEPTH_HEIGHT_RESOLUTION 	424
#define BASE_COLOR_WIDTH_RESOLUTION		1920
#define BASE_COLOR_HEIGHT_RESOLUTION	1080

#define COLOR_WIDTH_FOV					84.1
#define COLOR_HEIGHT_FOV				53.8
#define DEPTH_WIDTH_FOV					70.6
#define DEPTH_HEIGHT_FOV				60

#define MAX_ACC_RANGE					4000
#define MIN_ACC_RANGE					800

namespace vision_processing_color_seg {

	class Vision_Processor {
		private:
			ImageConverter _ic;
			vector<Color_Segmentation*> _frame_segmentations;
			Cups _msg;
			ros::Timer _timer;
			Mat _color_real_depth_mapping;
			camera_parameters _cam_coeficients;
			ofstream _log_file;

			virtual Mat depth_color_mapping(Mat color_frame, Mat depth_frame);
			virtual Point3f get_cup_pos(float color_x_coord, float color_y_coord);
			virtual bool is_point_valid(Point2f ptr, Mat frame);
			virtual Point3f depth_to_color_ptr(unsigned int depth, Point2f depth_coords);

		public:

			Vision_Processor() {}

			Vision_Processor(string camera_coeficients_name, string sub_topic) : _ic(sub_topic),
					_cam_coeficients(camera_coeficients_name) {

				_log_file.open("vision_log.txt");
			}

			Vision_Processor(string camera_coeficients_name, string sub_color_topic, string sub_depth_topic) :
				_ic(sub_color_topic, sub_depth_topic), _cam_coeficients(camera_coeficients_name) {

				_log_file.open("vision_log.txt");
			}

			Vision_Processor(string camera_coeficients_name, vector<Color_Segmentation*> frame_segmentations) :
				_frame_segmentations(frame_segmentations), _cam_coeficients(camera_coeficients_name) {

				_log_file.open("vision_log.txt");
			}

			Vision_Processor(string camera_coeficients_name, string sub_color_topic, string sub_depth_topic,
								vector<Color_Segmentation*> frame_segmentations) :
				_ic(sub_color_topic, sub_depth_topic), _frame_segmentations(frame_segmentations),
				_cam_coeficients(camera_coeficients_name) {

				_log_file.open("vision_log.txt");
			}

			virtual ~Vision_Processor();

			inline vector<Color_Segmentation*> get_frame_segmentations() const {
				return _frame_segmentations;
			}

			inline ImageConverter get_ic() const {
				return _ic;
			}

			inline ros::Timer get_timer() const {
				return _timer;
			}

			inline void set_frame_segmentations(vector<Color_Segmentation*> frameSegmentations) {
				_frame_segmentations = frameSegmentations;
			}

			inline void set_ic(ImageConverter ic) {
				_ic = ic;
			}

			inline void add_segmentation(Color_Segmentation* segmentation){
				_frame_segmentations.push_back(segmentation);
			}

			inline void set_feed_subscribe_topic(string sub_topic){
				_ic.assign_sub_color_topic(sub_topic);
			}

			inline void set_cups_publish_topic(string pub_topic){
				_ic.assign_pub_topic<robot_serving::Cups>(pub_topic);
			}

			inline void set_timer(ros::Timer timer) {
				_timer = timer;
			}

			bool infinite_center_mass(Point2f center_mass){
				return center_mass.x == INFINITY || center_mass.y == INFINITY;
			}

			void stop_timer(){
				_timer.stop();
			}

			virtual int image_processing();
			virtual void process_and_send(const TimerEvent& event);
			virtual void send_msg();
			virtual void clear_msg();

	};

}  // namespace vision_processing_color_seg

#endif /* VISION_PROCESSOR_H_ */

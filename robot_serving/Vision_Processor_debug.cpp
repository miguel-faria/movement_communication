/*
 * Vision_Processor.cpp
 *
 *  Created on: May 2, 2016
 *      Author: miguel
 */

#include "Vision_Processor.h"

namespace vision_processing_color_seg{

	Vision_Processor::~Vision_Processor(){
		_ic.get_image_color_sub().shutdown();
		_ic.get_image_sub_depth().shutdown();
		_ic.get_out_pub().shutdown();
		for (int var = 0; var < _frame_segmentations.size(); var++) {
					delete _frame_segmentations[var];
		}
		_frame_segmentations.erase(_frame_segmentations.begin(), _frame_segmentations.end());
		clear_msg();
	}

	bool Vision_Processor::is_point_valid(Point2f ptr, Mat frame){

		if(frame.empty())
			return false;

		return (ptr.x >= 0 && ptr.x < frame.cols && ptr.y >= 0 && ptr.y < frame.rows);

	}

	/*bool Vision_Processor::is_color_point_valid(Point2f ptr, unsigned int depth){


	}*/

	Point3f Vision_Processor::depth_to_color_ptr(unsigned int depth, Point2f depth_coords){

		Point3f color_coords, world_depth_coords, world_color_coords;

		world_depth_coords.x = ((depth_coords.x - _cam_coeficients.cx_depth) * (float(depth) / 1000)) /
				_cam_coeficients.fx_depth;
		world_depth_coords.y = ((depth_coords.y - _cam_coeficients.cy_depth) * (float(depth) / 1000)) /
				_cam_coeficients.fy_depth;
		world_depth_coords.z = float(depth) / 1000;

		world_color_coords = world_color_coords = _cam_coeficients.rotation_matrix * world_depth_coords
				+ Point3f(_cam_coeficients.translation_matrix(0),
						_cam_coeficients.translation_matrix(1),
						_cam_coeficients.translation_matrix(2));

		color_coords.x = ((world_color_coords.x * _cam_coeficients.fx_color) / world_color_coords.z) +
				_cam_coeficients.cx_color;
		color_coords.y = ((world_color_coords.y * _cam_coeficients.fy_color) / world_color_coords.z) +
				_cam_coeficients.cy_color;
		color_coords.z = (unsigned int)((world_color_coords.z - _cam_coeficients.translation_matrix(2)) * 1000);

		return color_coords;

	}

	Mat Vision_Processor::depth_color_mapping(Mat color_frame, Mat depth_frame){

		if(color_frame.empty()){
			cout << "No color frame received!" << endl;
			return Mat();
		}

		if(depth_frame.empty()){
			cout << "No depth frame received!" << endl;
			return Mat();
		}

		//FIXME: check if it is possible to know if a frame is already rectified!!!!

		stringstream ss;
		Mat world_color_mapping = Mat::ones(Size(depth_frame.cols, depth_frame.rows), depth_frame.depth()) * (MAX_DEPTH + 1);
		Point3f color_coords, color_coords_test;
		/*float depth_width_scale_base_res = float(BASE_DEPTH_WIDTH_RESOLUTION) / float(depth_frame.cols);
		float depth_heigth_scale_base_res = float(BASE_DEPTH_HEIGHT_RESOLUTION) / float(depth_frame.rows);
		float color_width_scale_base_res = float(BASE_COLOR_WIDTH_RESOLUTION) / float(color_frame.cols);
		float color_heigth_scale_base_res = float(BASE_COLOR_HEIGHT_RESOLUTION) / float(color_frinstame.rows);*/
		float width_down_scale = float(world_color_mapping.cols) / float(color_frame.cols);
		float height_down_scale = float(world_color_mapping.rows) / float(color_frame.rows);
		unsigned int depth, depth_color;

		for (int i = 0; i < depth_frame.cols; i++) {
			for (int j = 0; j < depth_frame.rows; ++j) {
				depth = depth_frame.at<uint16_t>(j,i);
				if(depth <= MAX_DEPTH)
					depth -= 300;
				color_coords = depth_to_color_ptr(depth, Point2f(i,j));
				depth_color = world_color_mapping.at<uint16_t>(round(color_coords.y * height_down_scale),
						round(color_coords.x * width_down_scale));
				if(is_point_valid(Point2f(round(color_coords.x * width_down_scale),
						round(color_coords.y * height_down_scale)), world_color_mapping)){
					if(depth_color > MAX_DEPTH ||
							(depth_color <= MAX_ACC_RANGE && color_coords.z < depth_color &&
									color_coords.z >= MIN_ACC_RANGE)){
						world_color_mapping.at<uint16_t>(round(color_coords.y * height_down_scale),
								round(color_coords.x * width_down_scale)) = color_coords.z + 300;
					}
				}
			}
		}

		return world_color_mapping;
	}

	Point3f Vision_Processor::get_cup_pos(float color_x_coord, float color_y_coord, Mat depth_cm){

		Point3f world_pos;
		unsigned int depth = _color_real_depth_mapping.at<uint16_t>(round(color_y_coord /_ic.get_ratio_height()),
				round(color_x_coord / _ic.get_ratio_width()));
		circle(depth_cm,
				Point2d(round(color_x_coord / _ic.get_ratio_width()), round(color_y_coord /_ic.get_ratio_height())),
				2, Scalar(65535, 65535, 65535), 2);
		imshow("Depth CM", depth_cm);
		if(depth > MAX_DEPTH){
			world_pos.x = world_pos.y = world_pos.z = INFINITY;
		}else {
			world_pos.x = ((color_x_coord - _cam_coeficients.cx_color) * (float(depth)/1000)) / _cam_coeficients.fx_color;
			world_pos.y = (-(color_y_coord - _cam_coeficients.cy_color) * (float(depth)/1000)) / _cam_coeficients.fy_color;
			world_pos.z = float(depth) / 1000;
		}

		return world_pos;
	}

	int Vision_Processor::image_processing(){
		Mat current_color_frame;
		Mat current_depth_frame;
		Mat color_cm;
		Mat depth_cm;
		int n_segmentations = _frame_segmentations.size();
		Point2f center_of_mass;
		string seg_color;
		vector<string> cups_color;
		vector<float> pos_x;
		vector<float> pos_y;
		vector<float> pos_z;
		float current_z;
		Point3f cup_pos;
		stringstream ss;
		ofstream out_file;

		_ic.get_color_frame().copyTo(current_color_frame);
		_ic.get_depth_frame().copyTo(current_depth_frame);

		if(_ic.get_out_pub().getTopic() == string()){
			ROS_ERROR("[VISION PROCESSING]: Publisher not associated to topic!");
			return -1;
		}

		if(current_color_frame.empty()){
			ROS_ERROR("[VISION PROCESSING]: No Color Frame received!");
			if(_ic.get_image_color_sub().getNumPublishers() < 1)
				ROS_INFO("[VISION PROCESSING]: No Color Frame publishers!");
			return -1;
		}

		if(current_depth_frame.empty()){
			ROS_ERROR("[VISION PROCESSING]: No Depth Frame received!");
			if(_ic.get_image_sub_depth().getNumPublishers() < 1)
				ROS_INFO("[VISION PROCESSING]: No Depth Frame publishers!");
			return -1;
		}

		depth_color_mapping(current_color_frame, current_depth_frame).copyTo(_color_real_depth_mapping);
		imshow("Color Mapped", _color_real_depth_mapping);

		current_color_frame.copyTo(color_cm);
		_color_real_depth_mapping.copyTo(depth_cm);

		for (int i = 0; i < n_segmentations; i++) {
			_frame_segmentations[i]->segmentation(current_color_frame);
			_frame_segmentations[i]->calculate_areas_centers_mass();
			seg_color = _frame_segmentations[i]->get_segment()->get_color();
			cups_color.push_back(seg_color);
			center_of_mass =_frame_segmentations[i]->get_segment()->get_bigger_cm();
			if(infinite_center_mass(center_of_mass)){
				pos_x.push_back(INFINITY);
				pos_y.push_back(INFINITY);
				pos_z.push_back(INFINITY);
			} else{
				cup_pos = get_cup_pos(center_of_mass.x, center_of_mass.y, depth_cm);
				pos_x.push_back(cup_pos.x);
				pos_y.push_back(cup_pos.y);
				pos_z.push_back(cup_pos.z);
			}
			imshow(seg_color, _frame_segmentations[i]->get_segment()->get_segmentation_map());
			circle(color_cm, center_of_mass, 3, Scalar(255,255,255), 2);
		}

		imshow("Color CM", color_cm);


		_msg.n_cups = n_segmentations;
		_msg.cups_color = cups_color;
		_msg.cups_pos_x = pos_x;
		_msg.cups_pos_y = pos_y;
		_msg.cups_pos_z = pos_z;
		_msg.depth_width = current_depth_frame.cols;
		_msg.depth_height = current_depth_frame.rows;

		_ic.set_color_frame(Mat());
		_ic.set_depth_frame(Mat());

		return 0;
	}

	void Vision_Processor::process_and_send(const TimerEvent& event){
		if(_timer.isValid()){
			if(image_processing() == 0){
				send_msg();
				clear_msg();
			}else {
				ROS_INFO("[VISION PROCESSING]: Message not sent");
				clear_msg();
			}
		}
	}

	void Vision_Processor::send_msg(){
		ROS_INFO_STREAM("[VISION PROCESSING]: Publishing cups position message to topic " << _ic.get_out_pub().getTopic());
		if(_ic.get_out_pub().getNumSubscribers() > 0){
			_ic.get_out_pub().publish(_msg);
			ROS_INFO("[VISION PROCESSING]: Publishing complete.");
			ROS_INFO_STREAM("[VISION PROCESSING]: " << _ic.get_out_pub().getNumSubscribers() <<
					" Subscribers in topic.\n\n");
		} else{
			ROS_INFO("[VISION PROCESSING]: Publishing incomplete.");
			ROS_INFO("[VISION PROCESSING]: No Subscribers in topic.\n\n");
		}

	}

	void Vision_Processor::clear_msg(){
		_msg.n_cups = 0;
		_msg.cups_color.clear();
		_msg.cups_pos_x.clear();
		_msg.cups_pos_y.clear();
		_msg.cups_pos_z.clear();
		_msg.depth_width = 0;
		_msg.depth_height = 0;
	}
}

/*
 * Camera_Parameters.h
 *
 *  Created on: May 5, 2016
 *      Author: miguel
 */

#ifndef VISION_PROCESSING_COLOR_SEG_CAMERA_PARAMETERS_H_
#define VISION_PROCESSING_COLOR_SEG_CAMERA_PARAMETERS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

namespace vision_processing_color_seg{

	struct camera_parameters {

			//Depth camera intrinsic parameters
			float cx_depth = 0.0f;
			float cy_depth = 0.0f;
			float fx_depth = 0.0f;
			float fy_depth = 0.0f;

			//Color camera intrisic parameters
			float cx_color = 0.0f;
			float cy_color = 0.0f;
			float fx_color = 0.0f;
			float fy_color = 0.0f;

			//Extrinsic parameters
			Matx33f rotation_matrix = Matx33f::zeros();
			Matx31f translation_matrix = Matx31f::zeros();

			camera_parameters(string param_file_name = ""){
				if(!param_file_name.empty()){
					string line, line2;
					ifstream param_file(param_file_name);
					istringstream iss1, iss2;
					string designation;
					float r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;

					if(!param_file.good()){
						ROS_ERROR("Camera_Parameters Constructor: Invalid File Name - File does not exist!!");
						exit(-1);
					}

					while(getline(param_file, line)){

						if(line.compare("Depth:") == 0){
 							getline(param_file, line);
							getline(param_file, line2);
							iss1.str(line);
							iss2.str(line2);
							if(!(iss1 >> designation >> fx_depth >> fy_depth)){
								cout << "Error Initializing Camera Depth Focal Parameters" << endl;
								return;
							}
							if(!(iss2 >> designation >> cx_depth >> cy_depth)){
								cout << "Error Initializing Camera Depth Central Parameters" << endl;
								return;
							}
							iss1.clear();
							iss2.clear();

						}else if(line.compare("Color:") == 0){
							getline(param_file, line);
							getline(param_file, line2);
							iss1.str(line);
							iss2.str(line2);
							if(!(iss1 >> designation >> fx_color >> fy_color)){
								cout << "Error Initializing Camera Color Focal Parameters" << endl;
								return;
							}
							if(!(iss2 >> designation >> cx_color>> cy_color)){
								cout << "Error Initializing Camera Color Center Parameters" << endl;
								return;
							}
							iss1.clear();
							iss2.clear();

						}else if(line.compare("Extrinsics:") == 0){
							getline(param_file, line);
							getline(param_file, line2);
							iss1.str(line);
							iss2.str(line2);

							iss1 >> designation;
							if(designation.compare("Translation:") == 0 || designation.compare("T:") == 0){
								iss1 >> t1 >> t2 >> t3;
								translation_matrix = Matx31f(t1, t2, t3);
								iss2 >> designation;
								iss2  >> r11 >> r12 >> r13 >> r21 >> r22 >> r23 >> r31 >> r32 >> r33;
								rotation_matrix = Matx33f(r11, r12, r13, r21, r22, r23, r31, r32, r33);
							}else if(designation.compare("Rotation:") == 0 || designation.compare("R:") == 0){
								iss1 >> r11 >> r12 >> r13 >> r21 >> r22 >> r23 >> r31 >> r32 >> r33;
								rotation_matrix = Matx33f(r11, r12, r13, r21, r22, r23, r31, r32, r33);
								iss2 >> designation;
								iss2 >> t1 >> t2 >> t3;
								translation_matrix = Matx31f(t1, t2, t3);
							}
							iss1.clear();
							iss2.clear();

						}else
							cout << "Ignoring file line" << endl;
					}
				}
			}
	};

}


#endif /* CAMERA_PARAMETERS_H_ */

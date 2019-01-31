/*
 * Color_Segment.h
 *
 *  Created on: Mar 24, 2016
 *      Author: miguel
 */

#ifndef COLOR_SEG_COLOR_SEGMENT_H_
#define COLOR_SEG_COLOR_SEGMENT_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

namespace color_seg {

	class Color_Segment {
	private:
			Mat _segmentation_map;
			vector<Point2f> *_centers_of_mass;
			vector<double> *_area_centers_of_mass;
			string _color;

	public:
			Color_Segment(string color) : _centers_of_mass(new vector<Point2f>()),
								_area_centers_of_mass(new vector<double>()), _color(color) {}

			Color_Segment(Mat segmentation_map = Mat(),
							vector<Point2f> *centers_of_mass = new vector<Point2f>(),
							vector<double> *areas = new vector<double>(),
							string color = "") :
				_segmentation_map(segmentation_map), _centers_of_mass(centers_of_mass), _area_centers_of_mass(areas),
				_color(color) {}

			~Color_Segment(){
				_centers_of_mass->erase(_centers_of_mass->begin(), _centers_of_mass->end());
				_area_centers_of_mass->erase(_area_centers_of_mass->begin(), _area_centers_of_mass->end());
				delete _centers_of_mass;
				delete _area_centers_of_mass;
			}

			inline vector<double>* get_area_centers_of_mass( ) const {
				return _area_centers_of_mass;
			}

			inline vector<Point2f>* get_centers_of_mass( ) const {
				return _centers_of_mass;
			}

			inline string get_color( ) const {
				return _color;
			}

			inline Mat get_segmentation_map( ) const {
				return _segmentation_map;
			}

			void set_area_centers_of_mass(vector<double>* areaCentersOfMass) {
				_area_centers_of_mass = areaCentersOfMass;
			}

			void set_centers_of_mass(vector<Point2f>* centersOfMass) {
				_centers_of_mass = centersOfMass;
			}

			void set_color(string color) {
				_color = color;
			}

			void set_segmentation_map(Mat segmentationMap) {
				_segmentation_map = segmentationMap;
			}

			Point2f get_bigger_cm(){
				if(_centers_of_mass->size() > 0){
					int dist = distance(_area_centers_of_mass->begin(),
							max_element(_area_centers_of_mass->begin(), _area_centers_of_mass->end()));
					if(dist < _centers_of_mass->size())
						return _centers_of_mass->at(dist);
					else
						return Point2f(INFINITY, INFINITY);
				}else{
					return Point2f(INFINITY, INFINITY);
				}
			}
	};

}  // namespace color_seg



#endif /* COLOR_SEG_COLOR_SEGMENT_H_ */

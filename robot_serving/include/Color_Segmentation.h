/*
 * Color_Segmentation.h
 *
 *  Created on: Mar 24, 2016
 *      Author: miguel
 */

#ifndef COLOR_SEG_COLOR_SEGMENTATION_H_
#define COLOR_SEG_COLOR_SEGMENTATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <cmath>
#include <string>
#include <vector>

#include "Color_Segment.h"


namespace color_seg {

	inline bool is_point(Point2f ptr){
		return (ptr.x == ptr.x) && (ptr.y == ptr.y);
	}

	class Color_Segmentation {
		private:
			Color_Segment* _segment;

		public:

			Color_Segmentation(string color = "") : _segment(new Color_Segment(color)) {}
			Color_Segmentation(Color_Segment* segment) : _segment(segment) {}

			virtual ~Color_Segmentation() {};

			inline Color_Segment* get_segment() const {
				return _segment;
			}

			void set_segment(Color_Segment* segment) {
				_segment = segment;
			}

			virtual void segmentation(Mat image) = 0;

			vector<Point2f>* compute_centers_of_mass(vector<vector<Point>> contours){
				vector<Point2f>* centers_of_mass = new vector<Point2f>();
				int contours_size = contours.size();
				Moments mu;
				Point2f ptr;

				for (int i = 0; i < contours_size; ++i) {
					mu = moments(contours[i], false);
					ptr = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);
					if(!isnan(ptr.x) && !isnan(ptr.y))
						centers_of_mass->push_back(ptr);
					else
						centers_of_mass->push_back(Point2f(INFINITY, INFINITY));
				}

				return centers_of_mass;
			}

			vector<double>* compute_areas(vector<vector<Point>> contours){
				vector<double>* areas = new vector<double>();
				int contours_size = contours.size();
				double area;

				for (int i = 0; i < contours_size; ++i) {
					area = contourArea(contours[i]);
					areas->push_back(area);
				}

				return areas;
			}

			void calculate_areas_centers_mass(){
				Mat store_img = _segment->get_segmentation_map();
				Mat canny_output;
				vector<vector<Point>> contours;
				vector<Vec4i> hierarchy;
				int thresh = 100;

				Canny( store_img, canny_output, thresh, thresh*2, 3 );
				findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

				_segment->set_area_centers_of_mass(compute_areas(contours));
				_segment->set_centers_of_mass(compute_centers_of_mass(contours));
			}
	};

	class HSV_Color_Segmentation : public Color_Segmentation {
		private:
			vector<int> _hue;
			vector<int> _saturation;
			vector<int> _value;

		public:

			HSV_Color_Segmentation(string color = "") : _hue(vector<int>()), _saturation(vector<int>()),
			_value(vector<int>()), Color_Segmentation(color) {}

			HSV_Color_Segmentation(Color_Segment* segment) : _hue(vector<int>()),
					_saturation(vector<int>()),_value(vector<int>()), Color_Segmentation(segment) {}

			HSV_Color_Segmentation(vector<int> hue, vector<int> saturation, vector<int> value, string color = "") :
				_hue(hue), _saturation(saturation), _value(value), Color_Segmentation(color) {}

			HSV_Color_Segmentation(vector<int> hue, vector<int> saturation, vector<int> value, Color_Segment* segment) :
				_hue(hue), _saturation(saturation), _value(value), Color_Segmentation(segment) {}

			~HSV_Color_Segmentation(){
				delete get_segment();
				_hue.erase(_hue.begin(), _hue.end());
				_saturation.erase(_saturation.begin(), _saturation.end());
				_value.erase(_value.begin(), _value.end());
			}

			inline vector<int> get_hue( ) const {
				return _hue;
			}

			inline vector<int> get_saturation( ) const {
				return _saturation;
			}

			inline vector<int> get_value( ) const {
				return _value;
			}

			inline void set_hue(const vector<int> hue) {
				_hue = hue;
			}

			inline void set_saturation(vector<int> saturation) {
				_saturation = saturation;
			}

			inline void set_value(vector<int> value) {
				_value = value;
			}

			void segmentation(Mat image){

				Mat img_out;
				cvtColor(image, img_out, COLOR_BGR2HSV);
				inRange(img_out, Scalar(_hue[0], _saturation[0], _value[0]),
						Scalar(_hue[1], _saturation[1], _value[1]), img_out);

				erode(img_out, img_out, getStructuringElement(MORPH_RECT, Size(5, 5)) );
				dilate(img_out, img_out, getStructuringElement(MORPH_RECT, Size(5, 5)) );

				dilate(img_out, img_out, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );
				erode(img_out, img_out, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );

				get_segment()->set_segmentation_map(img_out);
			}

	};

	class Red_HSV_Color_Segmentation : public HSV_Color_Segmentation {
		private:
			vector<int> _lower_hue;

		public:
			Red_HSV_Color_Segmentation() : _lower_hue(vector<int>()), HSV_Color_Segmentation("red") {}

			Red_HSV_Color_Segmentation(string color) : _lower_hue(vector<int>()), HSV_Color_Segmentation(color) {}

			Red_HSV_Color_Segmentation(Color_Segment* segment) : _lower_hue(vector<int>()),
					HSV_Color_Segmentation(segment) {}

			Red_HSV_Color_Segmentation(vector<int> lower_hue, vector<int> hue, vector<int> saturation,
					vector<int> value) : _lower_hue(lower_hue), HSV_Color_Segmentation(hue, saturation, value, "red") {}

			Red_HSV_Color_Segmentation(vector<int> lower_hue, vector<int> hue, vector<int> saturation,
					vector<int> value, string color) : _lower_hue(lower_hue),
							HSV_Color_Segmentation(hue, saturation, value, color) {}

			Red_HSV_Color_Segmentation(vector<int> lower_hue, vector<int> hue, vector<int> saturation,
					vector<int> value, Color_Segment *segment) : _lower_hue(lower_hue),
							HSV_Color_Segmentation(hue, saturation, value, segment) {}

			~Red_HSV_Color_Segmentation(){
				_lower_hue.erase(_lower_hue.begin(), _lower_hue.end());
			}

			inline vector<int> get_lower_hue( ) const {
				return _lower_hue;
			}

			inline void set_lower_hue(vector<int> lower_hue) {
				_lower_hue = lower_hue;
			}

			void segmentation(Mat image){

				Mat img_out, img_H, img_L, img_cvt;
				vector<int> hue = get_hue();
				vector<int> saturation = get_saturation();
				vector<int> value = get_value();
				cvtColor(image, img_cvt, COLOR_BGR2HSV);
				inRange(img_cvt, Scalar(hue[0], saturation[0], value[0]),
						Scalar(hue[1], saturation[1], value[1]), img_H);
				inRange(img_cvt, Scalar(_lower_hue[0], saturation[0], value[0]),
						Scalar(_lower_hue[1], saturation[1], value[1]), img_L);

				erode(img_H, img_H, getStructuringElement(MORPH_RECT, Size(5, 5)) );
				dilate(img_H, img_H, getStructuringElement(MORPH_RECT, Size(5, 5)) );
				dilate(img_H, img_H, getStructuringElement(MORPH_RECT, Size(5, 5)) );
				erode(img_H, img_H, getStructuringElement(MORPH_RECT, Size(5, 5)) );

				erode(img_L, img_L, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );
				dilate(img_L, img_L, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );
				dilate(img_L, img_L, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );
				erode(img_L, img_L, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );

				addWeighted(img_H, 1.0, img_L, 1.0, 0.0, img_out);

				get_segment()->set_segmentation_map(img_out);
			}
	};

}  // namespace color_seg



#endif /* COLOR_SEG_COLOR_SEGMENTATION_H_ */

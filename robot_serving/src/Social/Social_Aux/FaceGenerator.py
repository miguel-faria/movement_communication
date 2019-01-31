#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import abc, numpy, cv2


class FaceGenerator():
	__metaclass__ = abc.ABCMeta


	def __init__(self, screen_width = 1024, screen_height = 600):
		self._SCREEN_WIDTH = screen_width
		self._SCREEN_HEIGHT = screen_height

		self._face_image = []
		


	def get_generated_face(self):
		return self._face_image

	def draw_sinusoidal(self, amplitude, num_periods, offset, left_limit, right_limit, height,
						points_resolution = 300.0, color = (0,0,0), line_width = 10, fill = False):

		pts=[]

		for i in range(int(points_resolution)+1):

			pt=[int(left_limit+(i/points_resolution)*(right_limit-left_limit)),
			int(height+amplitude*numpy.sin(offset-1.57+i*((2*3.14*num_periods)/points_resolution)))]
			pts.append(pt)

		pts=[numpy.array(pts)]

		if not fill:
			cv2.polylines(self._face_image, pts, False, color, line_width, 8)
		else:
			cv2.fillPoly(self._face_image, pts, color, 8) 


	def draw_line(self, initial_point, final_point, width, color = (0,0,0)):
		cv2.line(self._face_image, initial_point, final_point, color, width, 8)

	def generate_circle_points(self, center, radius, start_angle, end_angle, number_samples = 100):
		angles = None
		if end_angle < 0:
			end_angle = end_angle + 360
			
		angles = numpy.linspace(end_angle, start_angle, number_samples, endpoint = True)
		
		list_points = []
		for angle in angles:
			list_points.append(tuple([int(center[0] + radius*numpy.cos(numpy.deg2rad(angle))), int(center[1] + radius*numpy.sin(numpy.deg2rad(-angle)))]))

		return numpy.array(list_points)



	@abc.abstractmethod
	def draw_eye_iris(self, left_eye_center = None, right_eye_center = None):
		raise NotImplemented("draw_eye_iris not implemented.")

	@abc.abstractmethod
	def draw_eye_orb(self, side = None):
		raise NotImplemented("draw_eye_orb not implemented.")

	@abc.abstractmethod
	def draw_face_skin(self):
		raise NotImplemented("draw_face_skin not implemented.")

	@abc.abstractmethod
	def happy_mouth(self):
		raise NotImplemented("happy_mouth not implemented.")

	@abc.abstractmethod
	def mega_happy_mouth(self):
		raise NotImplemented("mega_happy_mouth not implemented.")

	@abc.abstractmethod
	def sad_mouth(self):
		raise NotImplemented("sad_mouth not implemented.")

	@abc.abstractmethod
	def neutral_mouth(self):
		raise NotImplemented("neutral_mouth not implemented.")

	@abc.abstractmethod
	def wink_mouth(self, side):
		raise NotImplemented("wink_mouth not implemented.")

	@abc.abstractmethod
	def surprise_mouth(self):
		raise NotImplemented("surprise_mouth not implemented.")

	@abc.abstractmethod
	def afraid_mouth(self):
		raise NotImplemented("afraid_mouth not implemented.")

	@abc.abstractmethod
	def sleepy_eyebrows(self, amplitude = None, width = None, height_offset = None):
		raise NotImplemented("sleepy_eyebrows not implemented.")

	@abc.abstractmethod
	def neutral_eyebrows(self, height = None, length = None, eyebrow_width = None):
		raise NotImplemented("neutral_eyebrows not implemented.")

	@abc.abstractmethod
	def surprise_eyebrows(self, eyebrow_width = None): #DONE
		raise NotImplemented("surprise_eyebrows not implemented.")

	@abc.abstractmethod
	def delicate_eyebrows(self, eyebrow_width = None): #DONE
		raise NotImplemented("delicate_eyebrows not implemented.")


	@abc.abstractmethod
	def happy_eyebrows(self, eyebrow_width = None): #DONE
		raise NotImplemented("happy_eyebrows not implemented.")

	@abc.abstractmethod
	def wink_eyebrows(self, side, eyebrow_width = None): #DONE 
		raise NotImplemented("wink_eyebrows not implemented.")
	
	@abc.abstractmethod
	def _iris_normalizer(self, side, iris_position_to_look):
		raise NotImplemented("_iris_normalizer not implemented.")

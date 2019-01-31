#!/usr/bin/python2
# -*- coding: UTF-8 -*-


import numpy, cv2, rospy
from FaceGenerator import FaceGenerator


''' generated according to resolution 1024x600 '''
class BeaFace(FaceGenerator):
	def __init__(self, screen_width = 1024, screen_height = 600):
		FaceGenerator.__init__(self, screen_width, screen_height)
		self.__SKIN_COLOR = (13,49,212)
		self.__ORB_COLOR = (235, 235, 235)
		self.__EYELID_COLOR = (9, 39, 176)
		self.__IRIS_COLOR = (171, 86, 46)
		self.__EYE_BROW_COLOR = (0, 0, 0)
		self.__MOUTH_COLOR = (0, 0, 0)

		self.__ORB_RADIUS = 75*(self._SCREEN_WIDTH/1024)
		self.__IRIS_RADIUS = 25*(self._SCREEN_WIDTH/1024)

		self.__LEFT_EYE_CENTER = tuple([int(self._SCREEN_WIDTH*(1/4.0)),int(self._SCREEN_HEIGHT/2.5)])
		self.__RIGHT_EYE_CENTER = tuple([int(self._SCREEN_WIDTH*(3/4.0)),int(self._SCREEN_HEIGHT/2.5)])

		self.__awakening_start_time = rospy.get_rostime().secs


	def draw_face_skin(self):
		#third color channel
		self._face_image = numpy.zeros((self._SCREEN_HEIGHT, self._SCREEN_WIDTH, 3), numpy.uint8)
		self._face_image[:,:] = self.__SKIN_COLOR

	def happy_mouth(self): # DONE
		self.draw_sinusoidal(amplitude = 100, num_periods = 0.5, offset = 1.57, left_limit = self._SCREEN_WIDTH*(1/4.0),
							right_limit = self._SCREEN_WIDTH*(3/4.0), height = self._SCREEN_HEIGHT*(2/3.0), color = self.__MOUTH_COLOR)

	def mega_happy_mouth(self): # DONE
		self.draw_sinusoidal(amplitude = 150, num_periods = 0.5, offset = 1.57, left_limit = self._SCREEN_WIDTH*(1/4.0),
							right_limit = self._SCREEN_WIDTH*(3/4.0), height = self._SCREEN_HEIGHT*(2/3.0), fill = True, color = self.__MOUTH_COLOR)

	def sad_mouth(self): # DONE
		self.draw_sinusoidal(amplitude = -130, num_periods = 0.5, offset = 1.57, left_limit = self._SCREEN_WIDTH*(0.7/4.0),
							right_limit = self._SCREEN_WIDTH*(3.3/4.0), height = self._SCREEN_HEIGHT*(2.6/3.0), color = self.__MOUTH_COLOR)

	def neutral_mouth(self): # DONE
		self.draw_sinusoidal(amplitude = 0, num_periods = 2, offset = 0, left_limit = self._SCREEN_WIDTH*(1/4.0),
							right_limit = self._SCREEN_WIDTH*(3/4.0), height = self._SCREEN_HEIGHT * (2.2/3.0), color = self.__MOUTH_COLOR)

	def wink_mouth(self, side): # DONE
		if side == 'left':
			offset = -3.14
			left_limit = self._SCREEN_WIDTH*(2/4.0)
			right_limit = self._SCREEN_WIDTH*(3/4.0)
		else:
			offset = 1.57
			left_limit = self._SCREEN_WIDTH*(1/4.0)
			right_limit = self._SCREEN_WIDTH*(2/4.0)

		height = self._SCREEN_HEIGHT*(2/3.0)

		self.draw_sinusoidal(150, 0.25, offset, left_limit, right_limit, height, color = self.__MOUTH_COLOR)

	def surprise_mouth(self): # DONE
				
		center = (int(self._SCREEN_WIDTH/2.0),int(self._SCREEN_HEIGHT*(2/3.0)))
		mouth_radius = 50

		cv2.circle(self._face_image, center, mouth_radius, self.__MOUTH_COLOR, -1)

	def afraid_mouth(self): # DONE

		self.draw_sinusoidal(amplitude = 30.0 , num_periods = 3.0, offset = 0, left_limit = self._SCREEN_WIDTH*(1/4.0),
							right_limit = self._SCREEN_WIDTH*(3/4.0), height = self._SCREEN_HEIGHT*(2.2/3.0), color = self.__MOUTH_COLOR)

	def sleepy_eyebrows(self, amplitude = None, width = None, height_offset = None): #DONE
		if amplitude is None:
			amplitude = self._SCREEN_HEIGHT / 25.0
		
		if width is None:
			width = self._SCREEN_WIDTH / 10

		if height_offset is None:
			height_offset = 0

		left_pos = self.__LEFT_EYE_CENTER
		right_pos = self.__RIGHT_EYE_CENTER

		left_limit = left_pos[0] - width/2.0
		right_limit = left_pos[0] + width/2.0

		self.draw_sinusoidal(amplitude, 0.5, 1.57, left_limit, right_limit, left_pos[1]+height_offset, 300.0, self.__EYE_BROW_COLOR, 10)

		left_limit = right_pos[0] - width/2.0
		right_limit = right_pos[0] + width/2.0

		self.draw_sinusoidal(amplitude, 0.5, 1.57, left_limit, right_limit, right_pos[1]+height_offset, 300.0, self.__EYE_BROW_COLOR, 10)

	def neutral_eyebrows(self, height = None, length = None, eyebrow_width = None): #DONE
		if eyebrow_width is None:
			eyebrow_width = self._SCREEN_HEIGHT / 50

		if height is None:
			height = self._SCREEN_HEIGHT / 4.5

		if length is None:
			length = self._SCREEN_WIDTH / 5.5


		for center in [self.__LEFT_EYE_CENTER, self.__RIGHT_EYE_CENTER]:
			initial_point = tuple([int(center[0] - length/2.0), int(center[1] - height)])
			final_point = tuple([int(center[0] + length/2.0), int(center[1] - height)])

			self.draw_line(initial_point, final_point, eyebrow_width, self.__EYE_BROW_COLOR)

	def surprise_eyebrows(self, eyebrow_width = None): #DONE
		if eyebrow_width is None:
			eyebrow_width = self._SCREEN_HEIGHT / 50

		for  i,center in enumerate([self.__LEFT_EYE_CENTER, self.__RIGHT_EYE_CENTER]):
			initial_point = tuple([int(center[0]+pow(-1,i+1)*(1.75*self.__ORB_RADIUS)), center[1]])
			final_point = tuple([int(center[0]), int(center[1]-(2.0*self.__ORB_RADIUS))])

			self.draw_line(initial_point, final_point, eyebrow_width, self.__EYE_BROW_COLOR)

	def delicate_eyebrows(self, eyebrow_width = None): #DONE

		if eyebrow_width is None:
			eyebrow_width = self._SCREEN_HEIGHT / 50

		initial_point = (int(self.__RIGHT_EYE_CENTER[0]-(0.25*self.__ORB_RADIUS)),# moves upper sideways
		int(self.__RIGHT_EYE_CENTER[1]-(2.25*self.__ORB_RADIUS))) # moves upper downwards
		final_point = (int(self.__RIGHT_EYE_CENTER[0]+(1.5*self.__ORB_RADIUS)), # moves downer sideways
		int(self.__RIGHT_EYE_CENTER[1]-(1.0*self.__ORB_RADIUS))) # moves downer downwards

		self.draw_line(initial_point, final_point, eyebrow_width, self.__EYE_BROW_COLOR)

		initial_point = (int(self.__LEFT_EYE_CENTER[0]-(1.5*self.__ORB_RADIUS)),
		int(self.__LEFT_EYE_CENTER[1]-(1.25*self.__ORB_RADIUS)))
		final_point = (int(self.__LEFT_EYE_CENTER[0]+(0.0*self.__ORB_RADIUS)),
		int(self.__LEFT_EYE_CENTER[1]-(2.25*self.__ORB_RADIUS)))

		self.draw_line(initial_point, final_point, eyebrow_width, self.__EYE_BROW_COLOR)

		initial_point = final_point
		final_point = (int(self.__LEFT_EYE_CENTER[0]+(0.5*self.__ORB_RADIUS)),
		int(self.__LEFT_EYE_CENTER[1]-(1.75*self.__ORB_RADIUS)))
		
		self.draw_line(initial_point, final_point, eyebrow_width, self.__EYE_BROW_COLOR)

	def happy_eyebrows(self, eyebrow_width = None): #DONE

		if eyebrow_width is None:
			eyebrow_width = self._SCREEN_HEIGHT / 50

		for  i,center in enumerate([self.__LEFT_EYE_CENTER, self.__RIGHT_EYE_CENTER]):

			initial_point = (int(center[0]-pow(-1,i+1)*(0.4*self.__ORB_RADIUS)), # moves upper sideways
			int(center[1]-(1.7*self.__ORB_RADIUS)))  # moves upper downwards
			final_point = (int(center[0]+pow(-1,i+1)*(1.2*self.__ORB_RADIUS)), # moves downer sideways
			int(center[1]-(1.2*self.__ORB_RADIUS))) # moves downer downwards

			self.draw_line(initial_point, final_point, eyebrow_width, self.__EYE_BROW_COLOR)

	def wink_eyebrows(self, side, eyebrow_width = None): #DONE 
		if eyebrow_width is None:
			eyebrow_width = self._SCREEN_HEIGHT / 50

		if side == "left":
			wink_center=[self.__RIGHT_EYE_CENTER[0]-150,self.__RIGHT_EYE_CENTER[1]]
			eye_center=self.__LEFT_EYE_CENTER
			i = 0
			
		else:
			wink_center=[self.__LEFT_EYE_CENTER[0]+150,self.__LEFT_EYE_CENTER[1]]
			eye_center=self.__RIGHT_EYE_CENTER
			i = 1
			
		wink_coor=[125,-75]
		pt1=(int(wink_center[0]),int(wink_center[1]))
		pt2=(pt1[0]+pow(-1,i)*wink_coor[0],pt1[1]+wink_coor[1])
		
		self.draw_line(pt1, pt2, 20, (0,0,0))


		pt1 = (int(eye_center[0]+pow(-1,i+1)*(1.75*self.__ORB_RADIUS)),
			int(eye_center[1]-(0.75*self.__ORB_RADIUS)))
		pt2 = (int(eye_center[0]+pow(-1,i+1)*(0.75*self.__ORB_RADIUS)),
			int(eye_center[1]-(2.0*self.__ORB_RADIUS)))

		self.draw_line(pt1, pt2, eyebrow_width, self.__EYE_BROW_COLOR)

	def draw_eye_iris(self, left_eye_center = None, right_eye_center = None):
		if left_eye_center is None:
			left_eye_center = self.__LEFT_EYE_CENTER
		if right_eye_center is None:
			right_eye_center = self.__RIGHT_EYE_CENTER

		if left_eye_center == 0:
			cv2.circle(self._face_image, right_eye_center, self.__IRIS_RADIUS, self.__IRIS_COLOR, -1, 8)
			return

		if right_eye_center == 0:
			cv2.circle(self._face_image, left_eye_center, self.__IRIS_RADIUS, self.__IRIS_COLOR, -1, 8)			
			return

		cv2.circle(self._face_image, left_eye_center, self.__IRIS_RADIUS, self.__IRIS_COLOR, -1, 8)
		cv2.circle(self._face_image, right_eye_center, self.__IRIS_RADIUS, self.__IRIS_COLOR, -1, 8)
		

	def draw_eye_orb(self, side = None):
		if side is None:
			cv2.circle(self._face_image, self.__LEFT_EYE_CENTER, self.__ORB_RADIUS, self.__ORB_COLOR, -1, 8)
			cv2.circle(self._face_image, self.__RIGHT_EYE_CENTER, self.__ORB_RADIUS, self.__ORB_COLOR, -1, 8)
		elif side == 'left':
			cv2.circle(self._face_image, self.__LEFT_EYE_CENTER, self.__ORB_RADIUS, self.__ORB_COLOR, -1, 8)
		elif side == 'right':
			cv2.circle(self._face_image, self.__RIGHT_EYE_CENTER, self.__ORB_RADIUS, self.__ORB_COLOR, -1, 8)

	def draw_eyelid(self, bottom_angle = 90, top_angle = 90, frequency = 0.2, side = None):

		amplitude = (top_angle - bottom_angle) / 2.0
		offset = (top_angle + bottom_angle) / 2.0

		now = rospy.get_rostime()
		current_time = now.secs + pow(10,-9) * now.nsecs - self.__awakening_start_time

		alfa = offset + amplitude*numpy.sin(current_time*2*numpy.pi*frequency)

		if side is None:
			cv2.fillConvexPoly(self._face_image, self.generate_circle_points(self.__LEFT_EYE_CENTER, self.__ORB_RADIUS, alfa, 180-alfa), self.__EYELID_COLOR)
			cv2.fillConvexPoly(self._face_image, self.generate_circle_points(self.__RIGHT_EYE_CENTER, self.__ORB_RADIUS, alfa, 180-alfa), self.__EYELID_COLOR)		
		elif side == 'left':
			cv2.fillConvexPoly(self._face_image, self.generate_circle_points(self.__LEFT_EYE_CENTER, self.__ORB_RADIUS, alfa, 180-alfa), self.__EYELID_COLOR)
		elif side == 'right':
			cv2.fillConvexPoly(self._face_image, self.generate_circle_points(self.__RIGHT_EYE_CENTER, self.__ORB_RADIUS, alfa, 180-alfa), self.__EYELID_COLOR)

	''' Normalize iris position if it goes way off '''
	def _iris_normalizer(self, side, iris_position_to_look):

		if side == 'left':
			eye_center = self.__LEFT_EYE_CENTER 
		elif side == 'right':
			eye_center = self.__RIGHT_EYE_CENTER
		else:
			return None

		dist = pow(iris_position_to_look[0], 2) + pow(iris_position_to_look[1], 2)
		dist = pow(dist, 0.5)
		if dist > self.__ORB_RADIUS - self.__IRIS_RADIUS:
			
			good_ratio = (self.__ORB_RADIUS - self.__IRIS_RADIUS)/dist
			pt2 = numpy.array([iris_position_to_look[0],iris_position_to_look[1]])
			new_pt = numpy.multiply(pt2, good_ratio)
			iris_position_to_look = tuple([int(new_pt[0]),int(new_pt[1])])

		iris_position_to_look = tuple([int(iris_position_to_look[0]+eye_center[0]),int(-iris_position_to_look[1]+eye_center[1])])



		return iris_position_to_look

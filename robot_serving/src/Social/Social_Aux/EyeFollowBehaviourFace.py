#!/usr/bin/python2
# -*- coding: UTF-8 -*-


import rospy, numpy, tf

from EyeFollowBehaviour import EyeFollowBehaviour
from gaips_msgs.msg import PerceivedFace
from tf import TransformListener


class EyeFollowBehaviourFace(EyeFollowBehaviour):
	def __init__(self, tf_listener):
		EyeFollowBehaviour.__init__(self, tf_listener)

		self.__perceived_face = PerceivedFace()

		self.__horizontal_eye_movement_gain = 1.0
		self.__vertical_eye_movement_gain = 1.0

		rospy.Subscriber("/bea/perceived_face", PerceivedFace, self.__process_perceived_face)
	
	def __process_perceived_face(self, data):

		self.__perceived_face = data

	def get_iris_direction(self):

		[trans_head, rot_head] = self._get_tf_link('/head')
		if trans_head is None:

			horizontal_displacement = 0
			vertical_displacement = 0

		else:
			head_height = trans_head[2]
			head_angle = tf.transformations.euler_from_quaternion(rot_head)
			head_angle = numpy.rad2deg(head_angle[2])

			side_angle = self.__perceived_face.face_horizontal_displacement * (10.0/100)

			horizontal_displacement = self.__horizontal_eye_movement_gain * side_angle

			vertical_angle = self.__perceived_face.face_vertical_displacement * (10.0/100)
			vertical_displacement = self.__vertical_eye_movement_gain * vertical_angle

		eye_center_direction = tuple([int(horizontal_displacement),int(vertical_displacement)])

		return eye_center_direction
	

	
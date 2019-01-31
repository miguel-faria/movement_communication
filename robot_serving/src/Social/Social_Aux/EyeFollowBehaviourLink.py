#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import rospy, numpy, tf

from EyeFollowBehaviour import EyeFollowBehaviour
from tf import TransformListener

class EyeFollowBehaviourLink(EyeFollowBehaviour):
	def __init__(self, tf_listener, link_name):
		EyeFollowBehaviour.__init__(self, tf_listener)	
		self.__link_name = link_name

	def get_iris_direction(self):
		[trans_head, rot_head] = self._get_tf_link('/head')
		[trans, rot] = self._get_tf_link(self.__link_name)

		if trans is None or trans_head is None:
			horizontal_displacement = 0
			vertical_displacement = 0
		else:

			head_height = trans_head[2]
			head_angle = tf.transformations.euler_from_quaternion(rot_head)
			head_angle = numpy.rad2deg(head_angle[2])

			side_angle = numpy.rad2deg(numpy.arctan2(trans[1],trans[0]))

			horizontal_displacement = self._horizontal_eye_movement_gain*(side_angle - head_angle)

			vertical_angle = numpy.rad2deg(numpy.arctan2(trans[2] - head_height,trans[0]))
			vertical_displacement = self._vertical_eye_movement_gain * vertical_angle

		eye_center_direction = tuple([int(horizontal_displacement),int(vertical_displacement)])

		return eye_center_direction
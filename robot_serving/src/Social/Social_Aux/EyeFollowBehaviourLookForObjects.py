#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import rospy, numpy
from EyeFollowBehaviour import EyeFollowBehaviour


class EyeFollowBehaviourLookForObjects(EyeFollowBehaviour):
	def __init__(self, tf_listener):
		EyeFollowBehaviour.__init__(self, tf_listener)
		self.__PERIOD = 3.0
		self.__AMPLITUDE = 40
		self.__VERTICAL_DISPLACEMENT = -30
		self.__start_time = rospy.get_rostime().secs #+ pow(10,-9)*now.nsecs

	def get_iris_direction(self):

		now = rospy.get_rostime()
		current_time = now.secs + pow(10,-9)*now.nsecs - self.__start_time

		horizontal_displacement = self.__AMPLITUDE * numpy.cos((2 * 3.14/self.__PERIOD) * current_time)
		
		return tuple([int(horizontal_displacement),int(self.__VERTICAL_DISPLACEMENT)])
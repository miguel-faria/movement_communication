#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import rospy, os, sys, cv2, random, numpy, tf, time, abc

class EyeFollowBehaviour():
	__metaclass__ = abc.ABCMeta


	def __init__(self, tf_listener):
		self._tf_listener = tf_listener
		self._vertical_eye_movement_gain = (100.0/90)
		self._horizontal_eye_movement_gain = (100.0/90)

	def _get_tf_link(self, link_name):

		try:
			(trans_link,rot_link) = self._tf_listener.lookupTransform('/base', link_name, rospy.Time(0))
		except Exception as e:
			rospy.loginfo("TF from " + link_name + " error: " + str(e))
			return (None, None)

		return (trans_link, rot_link)

	@abc.abstractmethod
	def get_iris_direction(self):
		raise NotImplemented("Please implement method get_iris_direction")
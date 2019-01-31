#!/usr/bin/python2
# -*- coding: UTF-8 -*-

from EyeFollowBehaviour import EyeFollowBehaviour


class EyeFollowBehaviourNoFollow(EyeFollowBehaviour):
	def __init__(self, tf_listener):
		EyeFollowBehaviour.__init__(self, tf_listener)

	def get_iris_direction(self):
		return tuple([int(0),int(0)])
	

	
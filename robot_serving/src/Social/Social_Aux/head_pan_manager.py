#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import rospy, os, sys, random, math, numpy, tf
from baxter_interface import head
from gaips_msgs.msg import Head_mode, PerceivedFace, FollowBehaviour

class HeadPanManager():

	def __init__(self):
		self.__bea_head = head.Head()

		self.__COMMAND_RATE = rospy.Rate(1)
		self.__CONTROL_RATE = rospy.Rate(20)
		self.__LEFT_OFFSET_ERROR = 0
		self.__RIGHT_OFFSET_ERROR = -0.2

		self.__current_head_mode = Head_mode()
		self.__current_head_mode.head_mode = Head_mode.DEFAULT
		self.__current_face_dist = 0

		self.__face_sub = rospy.Subscriber('/bea/head_mode', Head_mode, self.__process_head_mode)
		self.image_sub = rospy.Subscriber('/bea/perceived_face', PerceivedFace, self.__process_new_face)

		self._tf_listener = tf.TransformListener()

	# This stores the last perception, which contains important information, like the position of the detected human face
	def __process_head_mode(self, data):
		self.__current_head_mode = data

		
	def __process_new_face(self, data):
		self.__current_face_dist = data.face_horizontal_displacement

	def __getAngleFromCoord(self, dist):
  		return dist * 2.09439510239 / 1280

	def __getAngleFromLink(self, link_name):
		try:
			(trans, rot) = self._tf_listener.lookupTransform('/torso', link_name, rospy.Time(0))
		except Exception as e:
			rospy.logwarn(str(e))
			return 0

		angle = math.atan2(trans[1], trans[0])

		if math.fabs(angle) > math.pi / 2.2:
			angle = math.copysign(math.pi / 2, angle) # Return first arg with the sign of second arg. 
		
		return angle

	def cycle(self):
		mode = self.__current_head_mode.head_mode

		# rospy.loginfo(mode)
		if mode == Head_mode.DEFAULT:
			# rospy.loginfo("Setting up head_mode to: default" )
			self.reset()
		elif mode == Head_mode.NOD_YES:
			# rospy.loginfo("Setting up head_mode to: nod_yes" )
			self.nod_yes()
			self.__current_head_mode.head_mode = Head_mode.DEFAULT
		elif mode == Head_mode.NOD_NO:
			# rospy.loginfo("Setting up head_mode to: nod_no" )
			self.nod_no()
			self.__current_head_mode.head_mode = Head_mode.DEFAULT
		elif mode == Head_mode.FOLLOW_BEHAVIOUR:
			behaviour = self.__current_head_mode.follow_behaviour
			
			follow_rule = behaviour.follow_rule
			if follow_rule == FollowBehaviour.FOLLOW_FACES:
				face_angle = self.__getAngleFromCoord(self.__current_face_dist)
				self.set_pan(face_angle)
			elif follow_rule == FollowBehaviour.FOLLOW_LINK:
				link_name = behaviour.name
				link_angle = self.__getAngleFromLink(link_name)
				self.set_pan(link_angle)

			else:
				rospy.logerr("Unknown Follow behaviour (= " + str(behaviour.follow_rule) + "). Moving to default")	
				self.__current_head_mode.head_mode = Head_mode.DEFAULT	

		else:
			rospy.logerr("Unknown head mode. Moving to default")
			self.__current_head_mode.head_mode = Head_mode.DEFAULT

	def nod_no(self, speed = 50, number_samples = 10, max_angle_wooble = 3.1415926535/6, number_times = 2):
		self.reset()
		
		angle_range = 2*max_angle_wooble
		sample_rate = angle_range/number_samples

		angles = numpy.linspace(-max_angle_wooble+self.__LEFT_OFFSET_ERROR, max_angle_wooble+self.__RIGHT_OFFSET_ERROR, number_samples, endpoint=True)
		
		self.__bea_head.set_pan(-max_angle_wooble, speed, timeout=0)
		self.__CONTROL_RATE.sleep()

		count = 0
		while count != number_times:
			count += 1
			for angle in angles:
				while not self.__bea_head.panning():
					self.__bea_head.set_pan(angle, speed, timeout=0)
				
				self.__CONTROL_RATE.sleep()

	def nod_yes(self, number_times = 2):
		self.reset()		
		count = 0
		while count != number_times:
			if not self.__bea_head.nodding():
				count += 1
				self.__bea_head.command_nod()

	def set_pan(self, angle):
		self.__bea_head.set_pan(angle)

	def reset(self):
		self.__bea_head.set_pan(0)

def main():
	rospy.init_node('bea_head_manager')
	head_manager = HeadPanManager()
	
	rospy.loginfo("Starting Head Manager")
	r = rospy.Rate(25)
	while not rospy.is_shutdown():
		head_manager.cycle()
		r.sleep()

if __name__ == "__main__":
	main()
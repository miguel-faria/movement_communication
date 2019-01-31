#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import rospy
import tf
from cv_bridge import CvBridge
from robot_serving.msg import FollowBehaviour, screen_mode
from sensor_msgs.msg import Image

from Social_Aux.BeaFace import BeaFace
from Social_Aux.EyeFollowBehaviourFace import EyeFollowBehaviourFace
from Social_Aux.EyeFollowBehaviourLink import EyeFollowBehaviourLink
from Social_Aux.EyeFollowBehaviourLookForObjects import EyeFollowBehaviourLookForObjects
from Social_Aux.EyeFollowBehaviourNoFollow import EyeFollowBehaviourNoFollow


class PerceptionReactor():

	def __init__(self):

		#list of possible perceptions
		self.__bridge = CvBridge()
		self.__image_pub = rospy.Publisher('/robot/xdisplay', Image, latch = True)

		self._tf_listener = None

		self.__face_generator = BeaFace()
		self.__face_follower = None
		self.__left_gripper_follower = None
		self.__right_gripper_follower = None
		self.__find_objects_follower = None
		self.__no_follower = None
		
		self.__current_screen_mode = screen_mode.HAPPY
		self.__perceived_face = None
		self.__follower_mode = None

		#rospy.Subscriber("/bea/screen_display_mode", screen_mode, self.__screen_mode)
		#rospy.Subscriber("/bea/follow_mode", FollowBehaviour, self.__follow_mode)
		rospy.loginfo("Starting perception reactor")

	# This stores the last perception, which contains important information, like the position of the detected human face
	def __process_perceived_face(self, data):
		self.__perceived_face = data

	# This stores the last desired screen display mode
	def __screen_mode(self, data):
		self.__current_screen_mode = data.screen_mode

	# This stores the last desired display mode
	def __follow_mode(self, data):
		follow_mode = data
		follow_rule = follow_mode.follow_rule

		if follow_rule == FollowBehaviour.DEFAULT:
			self.__follower_mode = self.__no_follower
		elif follow_rule == FollowBehaviour.FOLLOW_FACES:
			self.__follower_mode = self.__face_follower
		elif follow_rule == FollowBehaviour.FOLLOW_LINK:
			link_name = data.name
			if link_name == FollowBehaviour.LEFT_GRIPPER:
				self.__follower_mode = self.__left_gripper_follower
			elif link_name == FollowBehaviour.RIGHT_GRIPPER:
				self.__follower_mode = self.__right_gripper_follower
			else:
				raise Exception("PERCEPTIONS: NO LINK NAME PROVIDED IN FOLLOW_MODE")

		elif follow_rule == FollowBehaviour.FOLLOW_RANDOM_LOOK:
			self.__follower_mode = self.__find_objects_follower

		elif follow_rule == FollowBehaviour.FOLLOW_OBJECT:
			object_name = data.name
			rospy.logwarn("FOLLOW_OBJECT NOT IMPLEMENTED IN FOLLOW BEHAVIOUR")
			#Not implemented
			#Not implemented
			#Not implemented
		else:
			rospy.logwarn("Incorrect follow rule (= " + str(follow_rule) + "). Default mode DEFAULT chosen")
			self.__follower_mode = self.__no_follower

		self.cycle()

	def screen_mode(self, mode):
		self.__current_screen_mode = mode

	def follow_mode(self, mode, link):

		if mode == FollowBehaviour.FOLLOW_LINK:
			if link == 'left':
				self.__follower_mode = self.__left_gripper_follower
				return 1
			elif link == 'right':
				self.__follower_mode = self.__right_gripper_follower
				return 1
			else:
				rospy.loginfo('NO VALID LINK PROVIDED')
				return 0

		elif mode == FollowBehaviour.FOLLOW_RANDOM_LOOK:
			self.__follower_mode = self.__find_objects_follower

		else:
			rospy.loginfo("Incorrect follow mode (= " + str(mode) + "). Default mode DEFAULT chosen")
			self.__follower_mode = self.__no_follower

	def cycle(self):

		self.__face_generator.draw_face_skin()

		if self.__current_screen_mode == screen_mode.SLEEP:
			self.sleep_face()
		elif self.__current_screen_mode == screen_mode.SLEEPY:
			self.sleepy_face()
		elif self.__current_screen_mode == screen_mode.SURPRISED:
			self.surprise_face()
		elif self.__current_screen_mode == screen_mode.NEUTRAL:
			self.object_finder()
		elif self.__current_screen_mode == screen_mode.NERVOUS:
			self.delicate_face()
		elif self.__current_screen_mode == screen_mode.HAPPY:
			self.happy_face()
		elif self.__current_screen_mode == screen_mode.VERY_HAPPY:
			self.mega_happy_face()
		elif self.__current_screen_mode == screen_mode.WINK_LEFT:
			self.wink_face('left')
		elif self.__current_screen_mode == screen_mode.WINK_RIGHT:
			self.wink_face('right')
		elif self.__current_screen_mode == screen_mode.SAD:
			self.sad_face()
		else:
			rospy.logerr("UNKNOWN FACE_MODE")
			self.sleepy_face()


		self.convert_and_publish_image()
		# cv2.imshow('face',self.__face_generator.get_generated_face())	
		# cv2.waitKey(3)


	def convert_and_publish_image(self):
		self.__image_pub.publish(self.__bridge.cv2_to_imgmsg(self.__face_generator.get_generated_face(), encoding = 'bgr8'))
	
	def sleep_face(self):
		self.clean_face()
		self.__face_generator.neutral_mouth()
		self.__face_generator.sleepy_eyebrows()
		
	def sleepy_face(self):
		self.__face_generator.neutral_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		
		self.__face_generator.draw_eyelid(bottom_angle = -15, top_angle = 10, frequency = 0.2)	

	def delicate_face(self):
		self.__face_generator.afraid_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		self.__face_generator.delicate_eyebrows()

		
	def surprise_face(self):
		self.__face_generator.surprise_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)

		self.__face_generator.surprise_eyebrows()

	def wink_face(self, side):

		self.__face_generator.wink_eyebrows(side)
		self.__face_generator.wink_mouth(side)

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		if side == "left":
			self.__face_generator.draw_eye_orb("left")
			self.__face_generator.draw_eye_iris(left_eye_position, 0)
		elif side == "right":
			self.__face_generator.draw_eye_orb("right")
			self.__face_generator.draw_eye_iris(0, right_eye_position)

	def object_finder(self):

		self.__face_generator.neutral_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		self.__face_generator.neutral_eyebrows()


	def happy_face(self):
		self.__face_generator.happy_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		self.__face_generator.happy_eyebrows()

	def mega_happy_face(self):
		self.__face_generator.mega_happy_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		self.__face_generator.neutral_eyebrows()
		# self.__face_generator.sleepy_eyebrows(-50,150,-100)

	def sad_face(self):
		self.__face_generator.sad_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)

	def angry_face(self):
		self.__face_generator.sad_mouth()
		self.__face_generator.draw_eye_orb()

		left_eye_position = self.__face_generator._iris_normalizer('left', self.__follower_mode.get_iris_direction())
		right_eye_position = self.__face_generator._iris_normalizer('right', self.__follower_mode.get_iris_direction())

		self.__face_generator.draw_eye_iris(left_eye_position, right_eye_position)
		self.__face_generator.neutral_eyebrows()

	def start(self):
		self._tf_listener = tf.TransformListener()
		self.__face_follower = EyeFollowBehaviourFace(self._tf_listener)
		self.__left_gripper_follower = EyeFollowBehaviourLink(self._tf_listener, '/left_gripper')
		self.__right_gripper_follower = EyeFollowBehaviourLink(self._tf_listener, '/right_gripper')
		self.__find_objects_follower = EyeFollowBehaviourLookForObjects(self._tf_listener)
		self.__no_follower = EyeFollowBehaviourNoFollow(self._tf_listener)
		self.__follower_mode = self.__no_follower
		self.__face_generator.draw_face_skin()

	def clean_face(self):
		self.__face_generator.draw_face_skin()

if __name__ == "__main__":
	rospy.init_node('bea_perceptions_listener')
	perception_reactor = PerceptionReactor()
	perception_reactor.start()

	r = rospy.Rate(50) # 50hz
	while not rospy.is_shutdown():
		perception_reactor.cycle()
		r.sleep()
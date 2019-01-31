from enum import Enum

import rospy
from sensor_msgs.msg import Image


class MovementState(Enum):
	pending = 0
	started = 1
	success = 2
	failure = 3


class FacialExpressionManager(object):

	def __init__(self, path):
		self.folder_path = '/' + path + '/'
		self.baxter_LCD_pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
		self.movement_state = MovementState.pending


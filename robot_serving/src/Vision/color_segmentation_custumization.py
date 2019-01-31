import argparse

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class color_image_processor(object):

	def __init__(self, color_topic):

		self._max_hue = 0
		self._max_saturation = 0
		self._max_value = 0
		self._min_hue = 0
		self._min_saturation = 0
		self._min_value = 0
		self._bridge = CvBridge()
		self._image = np.zeros((520, 520, 3), np.uint8)
		self._segmented_image = np.zeros((520, 520, 3), np.uint8)

		# Create a black image, a window
		cv2.namedWindow('image')
		cv2.namedWindow('segmented_image')

		# create trackbars for color change
		cv2.createTrackbar('Max_Hue', 'image', 0, 255, self.callback_sliders)
		cv2.createTrackbar('Min_Hue', 'image', 0, 255, self.callback_sliders)
		cv2.createTrackbar('Max_Saturation', 'image', 0, 255, self.callback_sliders)
		cv2.createTrackbar('Min_Saturation', 'image', 0, 255, self.callback_sliders)
		cv2.createTrackbar('Max_Value', 'image', 0, 255, self.callback_sliders)
		cv2.createTrackbar('Min_Value', 'image', 0, 255, self.callback_sliders)

		# create switch for ON/OFF functionality
		self._switch = '0 : OFF \n1 : ON'
		cv2.createTrackbar(self._switch, 'image', 0, 1, self.callback_sliders)
		cv2.waitKey(3)

		self._image_sub = rospy.Subscriber(color_topic, Image, self.receive_image)
		self._timer = rospy.Timer(rospy.Duration(0.25), self.segmentation)

	def receive_image(self, data):
		try:
			cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
			self._image = cv_image
		except CvBridgeError as e:
			print(e)

	def callback_sliders(self, x):
		rospy.loginfo('Trackbar Callback')
		self._max_hue = cv2.getTrackbarPos('Max_Hue', 'image')
		self._max_saturation = cv2.getTrackbarPos('Max_Saturation', 'image')
		self._max_value = cv2.getTrackbarPos('Max_Value', 'image')
		self._min_hue = cv2.getTrackbarPos('Min_Hue', 'image')
		self._min_saturation = cv2.getTrackbarPos('Min_Saturation', 'image')
		self._min_value = cv2.getTrackbarPos('Min_Value', 'image')

	def segmentation(self, event):
		s = cv2.getTrackbarPos(self._switch, 'image')
		self._segmented_image = np.zeros((520, 520, 3), np.uint8)
		cv2.imshow('image', self._image)
		cv2.waitKey(3)
		if s != 0:
			hsv_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
			lower_bound = np.array([self._min_hue, self._min_saturation, self._min_value])
			upper_bound = np.array([self._max_hue, self._max_saturation, self._max_value])
			mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
			kernel = np.ones((5, 5), np.uint8)

			self._segmented_image = cv2.erode(mask, kernel)
			self._segmented_image = cv2.dilate(self._segmented_image, kernel)
			self._segmented_image = cv2.dilate(self._segmented_image, kernel)
			self._segmented_image = cv2.erode(self._segmented_image, kernel)

			cv2.imshow('segmented_image', self._segmented_image)
			cv2.waitKey(3)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t',
						dest='topic',
						default='/kinect2/hd/image_color_rect',
						help='ROS topic from color images. (default: /kinect2/hd/image_color_rect)')
	args = parser.parse_args()

	rospy.init_node('cs_calibration', anonymous=True)
	ic = color_image_processor(args.topic)

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
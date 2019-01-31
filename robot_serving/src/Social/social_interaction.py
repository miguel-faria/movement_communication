import argparse
import rospy
import time
from std_msgs.msg import Int32

from Social_Aux.SocialInteraction import SocialInteraction


def main():
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-n',
						required=True,
						dest='n_interaction',
						help='Current interaction number')
	args = parser.parse_args()

	rospy.init_node("social_interaction_robot_serving")
	pub = rospy.Publisher('start', Int32, queue_size=100)
	interaction = SocialInteraction(int(args.n_interaction))
	rospy.on_shutdown(interaction.move_neutral)
	interaction.hello()

	rate = rospy.Rate(10)
	while not rospy.is_shutdown() and not interaction.over():
		pub.publish(1)
		rate.sleep()
		interaction.cycle()

	time.sleep(5)
	interaction.goodbye()
	interaction.close_file()

if __name__ == '__main__':
	main()
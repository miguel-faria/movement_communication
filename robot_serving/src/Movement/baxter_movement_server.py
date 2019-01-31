import argparse
import sys

import rospy

from Baxter_Movement_Aux.hybrid_traj_service import HybridTrajectoryServer
from Baxter_Movement_Aux.legible_traj_service import LegibleTrajectoryServer
from Baxter_Movement_Aux.predictable_traj_service import PredictTrajectoryServer
from std_srvs.srv import Empty, EmptyResponse

finish = False


def finish_signal_handler(req):
	global finish
	finish = True
	return EmptyResponse()


def main():
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-t',
						type=str,
						required=True,
						dest='traj_type',
						help='Type of trajectory to perform based on the three approaches, '
							 'valid values: predict, legible, hybrid')
	args = parser.parse_args()

	rospy.init_node('baxter_movement_server')

	if args.traj_type == 'predict':
		traj_server = PredictTrajectoryServer()
	elif args.traj_type == 'legible':
		traj_server = LegibleTrajectoryServer()
	elif args.traj_type == 'hybrid':
		traj_server = HybridTrajectoryServer()
	else:
		rospy.logerr('Invalid movement trajectory type! Exiting...')
		sys.exit(1)
	rospy.on_shutdown(traj_server.send_neutral)
	end_service = rospy.Service('finish_movement_server', Empty, finish_signal_handler)
	rospy.loginfo('Baxter Movement Server Initiated')

	rate = rospy.Rate(10)
	while not rospy.is_shutdown() and not finish:
		rate.sleep()

	traj_server.send_neutral()
	sys.exit(0)

if __name__ == '__main__':
	main()
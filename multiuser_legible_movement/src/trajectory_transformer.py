import pandas as pd
import numpy as np
import math
import tf.transformations as T
import csv

from user_perspective_legibility import UserPerspectiveLegibility
from geometry_msgs.msg import Pose, Point


def main():
	reader = pd.read_csv('../data/3_users_10000_1.csv')
	trajectories = {}
	for _, row in reader.iterrows():
		row_np = np.fromstring(row[1], dtype=float, sep=' ')
		trajectories[int(row[0])] = row_np.reshape((int(len(row_np) / 3), 3))

	user1_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(90), ak=math.radians(0), axes='rxyz')
	user2_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(0), ak=math.radians(0), axes='rxyz')
	user3_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(180), ak=math.radians(0), axes='rxyz')
	robot_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(0), ak=math.radians(180), axes='rxyz')

	user1_translation = (2.0, 3.75, 0.0)
	user2_translation = (5.5, 4.5, 0.0)
	user3_translation = (5.5, 3.0, 0.0)
	robot_translation = (15.0, 3.75, 0.0)

	user1_pose = Pose(position=user1_translation, orientation=user1_rot)
	user2_pose = Pose(position=user2_translation, orientation=user2_rot)
	user3_pose = Pose(position=user3_translation, orientation=user3_rot)
	robot_pose = Pose(position=robot_translation, orientation=robot_rot)

	user1 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user1',
	                                  user_pose=user1_pose, robot_pose=robot_pose)
	user2 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user2',
	                                  user_pose=user2_pose, robot_pose=robot_pose)
	user3 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user3',
	                                  user_pose=user3_pose, robot_pose=robot_pose)
	robot = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='robot',
	                                  user_pose=robot_pose, robot_pose=robot_pose)

	# user1_trajectories = {}
	# user2_trajectories = {}
	# user3_trajectories = {}
	dict_keys = trajectories.keys()
	writer_user1 = csv.writer(open("../data/3_users_10000_u1.csv", "w"))
	writer_user2 = csv.writer(open("../data/3_users_10000_u2.csv", "w"))
	writer_user3 = csv.writer(open("../data/3_users_10000_u3.csv", "w"))
	writer_robot = csv.writer(open("../data/3_users_10000_r.csv", "w"))

	writer_user1.writerow(['Iteration', 'Trajectory'])
	writer_user2.writerow(['Iteration', 'Trajectory'])
	writer_user3.writerow(['Iteration', 'Trajectory'])
	writer_robot.writerow(['Iteration', 'Trajectory'])
	for key in dict_keys:
		traj = trajectories[key]

		writer_user1.writerow([key, user1.transformTrajectory(traj)])
		writer_user2.writerow([key, user2.transformTrajectory(traj)])
		writer_user3.writerow([key, user3.transformTrajectory(traj)])
		writer_robot.writerow([key, robot.transformTrajectory(traj)])


if __name__ == '__main__':
	main()

#! /usr/bin/env python2

from __future__ import print_function
import rospy
import numpy as np
import math
import tf.transformations as T
import matplotlib.pyplot as plt
import csv
import sys, os

# from pathlib import Path
from user_perspective_legibility import UserPerspectiveLegibility
from legible_trajectory import LegibleMovement
from image_annotations_3d import ImageAnnotations3D
from geometry_msgs.msg import Pose, Quaternion, Point
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict


def main():

	file_path = os.path.dirname(sys.argv[0])
	full_path = os.path.abspath(file_path)
	image_dir = full_path + '/images'

	# Configuration 1
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Configuration 2
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(300), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(60), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Configuration 3
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(325), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(35), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Configuration 4
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(300), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(60), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Configuration 5
	# user1_rot = T.quaternion_from_euler(ai=math.radians(315), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(295), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Configuration 6
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(330), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(315), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Robot rotation
	# robot_rot = T.quaternion_from_euler(ai=math.radians(180), aj=math.radians(90), ak=math.radians(0), axes='rzxy')

	# User Orientation Simulation
	user1_rot = T.quaternion_from_euler(ai=math.radians(180), aj=math.radians(-10), ak=math.radians(0), axes='ryxz')
	user2_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(-10), ak=math.radians(0), axes='ryxz')
	user3_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(-10), ak=math.radians(0), axes='ryxz')
	# Robot Orientarion Simulation
	robot_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(0), ak=math.radians(0), axes='ryxz')

	# Configuration 1-4
	# user1_translation = (1500.0, 1000.0, 100.0)
	# user2_translation = (500.0, 1500.0, 100.0)
	# user3_translation = (2500.0, 1500.0, 100.0)
	# Configuration 5
	# user1_translation = (500.0, 1000.0, 100.0)
	# user2_translation = (500.0, 1500.0, 100.0)
	# user3_translation = (500.0, 2000.0, 100.0)
	# Configuration 6
	# user1_translation = (1500.0, 1000.0, 100.0)
	# user2_translation = (1000.0, 1200.0, 100.0)
	# user3_translation = (500.0, 1400.0, 100.0)
	# Robot Position
	# robot_translation = (1500.0, 2500.0, 100.0

	# User Position Simulation
	user1_translation = (1500.0, 1000.0, 500.0)
	user2_translation = (3200.0, 1000.0, 1750.0)
	user3_translation = (-250.0, 1000.0, 1750.0)
	# Robot Position Simulation
	robot_translation = (1500.0, 250.0, 2400.0)

	# Poses
	robot_pose = Pose(position=robot_translation, orientation=robot_rot)
	user1_pose = Pose(position=user1_translation, orientation=user1_rot)
	user2_pose = Pose(position=user2_translation, orientation=user2_rot)
	user3_pose = Pose(position=user3_translation, orientation=user3_rot)

	# Configuration 1-3
	# targets = {'A': np.array([1200.0, 1300.0, 100.0]), 'B': np.array([1500.0, 1300.0, 100.0]),
	#            'C': np.array([1800.0, 1300.0, 100.0])}
	# targets = {'A': np.array([1200.0, 1500.0, 100.0]), 'B': np.array([1500.0, 1500.0, 100.0]),
	#            'C': np.array([1800.0, 1500.0, 100.0])}
	# targets = {'A': np.array([1200.0, 2050.0, 100.0]), 'B': np.array([1500.0, 2050.0, 100.0]),
	#            'C': np.array([1800.0, 2050.0, 100.0])}
	# Configuration 4
	# targets = {'A': np.array([1400.0, 1600.0, 100.0]), 'B': np.array([1600.0, 1600.0, 100.0]),
	#            'C': np.array([1500.0, 1400.0, 100.0])}
	# Configuration 5
	# targets = {'A': np.array([1200.0, 1500.0, 100.0]), 'B': np.array([1500.0, 1500.0, 100.0]),
	#            'C': np.array([1800.0, 1500.0, 100.0])}
	# Configuration 6
	# targets = {'A': np.array([1250.0, 1700.0, 100.0]), 'B': np.array([1500.0, 1500.0, 100.0]),
	#            'C': np.array([1800.0, 1300.0, 100.0])}

	# Taeget Position Simulation
	# targets = {'A': np.array([1800.0, 250.0, 1700.0]), 'B': np.array([1500.0, 250.0, 1700.0]),
	#            'C': np.array([1200.0, 250.0, 1700.0])}
	targets = {'A': np.array([1650.0, 250.0, 1700.0]), 'B': np.array([1500.0, 250.0, 1500.0]),
			   'C': np.array([1300.0, 250.0, 1700.0])}
	# targets = {'A': np.array([1650.0, 250.0, 1700.0]), 'B': np.array([1500.0, 250.0, 1900.0]),
	#            'C': np.array([1300.0, 250.0, 1700.0])}
	# targets = {'A': np.array([1650.0, 250.0, 1700.0]), 'B': np.array([1500.0, 250.0, 1700.0]),
	#            'C': np.array([1350.0, 250.0, 1700.0])}
	# targets = {'A': np.array([1600.0, 250.0, 1800.0]), 'B': np.array([1600.0, 250.0, 1600.0]),
	#            'C': np.array([1400.0, 250.0, 1700.0])}
	# targets = {'A': np.array([1500.0, 250.0, 2000.0]), 'B': np.array([1500.0, 250.0, 1400.0]),
	#            'C': np.array([1500.0, 250.0, 1700.0])}
	# targets = {'A': np.array([1400, 250.0, 1800.0]), 'B': np.array([1400.0, 250.0, 1600.0]),
	#            'C': np.array([1600.0, 250.0, 1700.0])}

	# Parameterization
	n_users = 1
	n_targets = len(targets)
	optim_target = 'C'
	# user_poses = [user1_pose, user2_pose, user3_pose]
	user_poses = [user3_pose]
	# u_ids = ['user1', 'user2', 'user3']
	u_ids = ['user3']
	# filename = "../data/3_users_conf_3_c_sim1.csv"
	filename = "../data/1_user_conf_3_u3_c_sim1.csv"

	# Trajectory Creation
	# traj_x = np.linspace(robot_translation[0], targets[optim_target][0], num=20)[:, None]
	# traj_y = np.linspace(robot_translation[1], targets[optim_target][1], num=20)[:, None]
	# traj_z = np.linspace(targets[optim_target][2], targets[optim_target][2], num=20)[:, None]
	# Trajectory Creation Simulation
	traj_x = np.linspace(robot_translation[0], targets[optim_target][0], num=20)[:, None]
	traj_y = np.linspace(targets[optim_target][1], targets[optim_target][1], num=20)[:, None]
	traj_z = np.linspace(robot_translation[2], targets[optim_target][2], num=20)[:, None]
	traj_1 = np.hstack((traj_x, traj_y, traj_z))

	# Variables and definitions for performance evaluation
	user_poses_defined = [user1_pose, user2_pose, user3_pose]
	user_defined_ids = ['user1', 'user2', 'user3']

	# user1 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user1',
	#                                   user_pose=user1_pose, robot_pose=robot_pose)
	# user2 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user2',
	#                                   user_pose=user2_pose, robot_pose=robot_pose)
	# user3 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user3',
	#                                   user_pose=user3_pose, robot_pose=robot_pose)
	# robot = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='robot',
	#                                   user_pose=robot_pose, robot_pose=robot_pose)
	# Trajectory transformation testing
	# user1_traj = user1.transformTrajectory(traj_1, viewport=False)
	# user2_traj = user2.transformTrajectory(traj_1, viewport=False)
	# user3_traj = user3.transformTrajectory(traj_1, viewport=False)
	# robot_traj = robot.transformTrajectory(traj_1, viewport=False)
	# point_0 = np.array([0, 0, 0, 1])
	# point_1 = np.array([1, 1, 1, 1])
	# user1_ptr0 = np.linalg.inv(user1_transform).dot(point_0)
	# user1_ptr1 = np.linalg.inv(user1_transform).dot(point_1)

	# Position Transformation
	# user1_targets = user1.transformTrajectory(targets, viewport=False)
	# user1_robot = user1.transformTrajectory(np.array([robot_translation]), viewport=False)
	# user2_targets = user2.transformTrajectory(targets, viewport=False)
	# user2_robot = user2.transformTrajectory(np.array([robot_translation]), viewport=False)
	# user3_targets = user3.transformTrajectory(targets, viewport=False)
	# user3_robot = user3.transformTrajectory(np.array([robot_translation]), viewport=False)

	# print('Original Trajectory')
	# print(traj_1)
	# print(traj_2)
	# print('Robot transformed trajectory')
	# print(robot_traj)
	# print('User 1 transformed trajectory')
	# print(user1_traj)
	# print(user1_robot)
	# print(user1_traj / user1_traj[:, 2, None])
	# print(user1_traj_2)
	# print(user1_traj_2 / user1_traj_2[:, 2, None])
	# print('User 2 transformed trajectory')
	# print(user2_traj)
	# print(user2_traj / user2_traj[:, 2, None])
	# print(user2_traj_2)
	# print(user2_traj_2 / user2_traj_2[:, 2, None])
	# print('User 3 transformed trajectory')
	# print(user3_traj)
	# print(user3_traj / user3_traj[:, 2, None])
	# print(user3_traj_2)
	# print(user3_traj_2 / user3_traj_2[:, 2, None])

	# fig = plt.figure('Original Trajectory')
	# ax = fig.add_subplot(111, projection=Axes3D.name)
	# ax.plot(traj_1[:, 0], traj_1[:, 1], traj_1[:, 2], 'black', label='Trajectory', marker='.', linestyle="None")
	# ax.plot(np.array([user1_translation[0]]), np.array([user1_translation[1]]), np.array([user1_translation[2]]),
	#         color='red', marker='2', markersize=15, label='User1')
	# ax.plot(np.array([user2_translation[0]]), np.array([user2_translation[1]]), np.array([user2_translation[2]]),
	#         color='green', marker='2', markersize=15, label='User2')
	# ax.plot(np.array([user3_translation[0]]), np.array([user3_translation[1]]), np.array([user3_translation[2]]),
	#         color='brown', marker='2', markersize=15, label='User3')
	# ax.plot(np.array([robot_translation[0]]), np.array([robot_translation[1]]), np.array([robot_translation[2]]),
	#         color='blue', marker='2', markersize=15, label='Robot')
	# ax.plot(np.array([targets['A'][0]]), np.array([targets['A'][1]]), np.array([targets['A'][2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.plot(np.array([targets['B'][0]]), np.array([targets['B'][1]]), np.array([targets['B'][2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.plot(np.array([targets['C'][0]]), np.array([targets['C'][1]]), np.array([targets['C'][2]]),
	#         color='darkorange', marker='D', markersize=10)
	# plt.legend(loc='best')
	# ax.view_init(azim=180, elev=-10)
	# fig.show()

	# fig2 = plt.figure('User 1')
	# ax = fig2.gca(projection='3d')
	# ax.plot(user1_traj[:, 0], user1_traj[:, 1], user1_traj[:, 2], 'green')
	# ax.plot(np.array([0]), np.array([0]), np.array([0]), color='red', marker='2', markersize=15, label='User 1')
	# ax.plot(np.array([user1_robot[0, 0]]), np.array([user1_robot[0, 1]]), np.array([user1_robot[0, 2]]),
	#         color='blue', marker='2', markersize=15, label='Robot')
	# ax.plot(np.array([user1_targets[0, 0]]), np.array([user1_targets[0, 1]]), np.array([user1_targets[0, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.plot(np.array([user1_targets[1, 0]]), np.array([user1_targets[1, 1]]), np.array([user1_targets[1, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.set_xlim3d(-500, 500)
	# ax.set_ylim3d(-2000, 2000)
	# ax.set_zlim3d(-1500, 1500)
	# plt.legend(loc='best')
	# fig2.show()

	# fig3 = plt.figure('User 2')
	# ax = fig3.gca(projection='3d')
	# ax.plot(user2_traj[:, 0], user2_traj[:, 1], user2_traj[:, 2], 'blue')
	# ax.plot(np.array([0]), np.array([0]), np.array([0]), color='red', marker='2', markersize=15, label='User 2')
	# ax.plot(np.array([user2_robot[0, 0]]), np.array([user2_robot[0, 1]]), np.array([user2_robot[0, 2]]),
	#         color='blue', marker='2', markersize=15, label='Robot')
	# ax.plot(np.array([user2_targets[0, 0]]), np.array([user2_targets[0, 1]]), np.array([user2_targets[0, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.plot(np.array([user2_targets[1, 0]]), np.array([user2_targets[1, 1]]), np.array([user2_targets[1, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.plot(np.array([user2_targets[2, 0]]), np.array([user2_targets[2, 1]]), np.array([user2_targets[2, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.set_xlim3d(-500, 500)
	# ax.set_ylim3d(-2000, 2000)
	# ax.set_zlim3d(-1500, 1500)
	# plt.legend(loc='best')
	# ax.view_init(azim=235, elev=9)
	# fig3.show()

	# fig4 = plt.figure('User 3')
	# ax = fig4.gca(projection='3d')
	# ax.plot(user3_traj[:, 0], user3_traj[:, 1], user3_traj[:, 2], 'orange')
	# ax.plot(np.array([0]), np.array([0]), np.array([0]), color='red', marker='2', markersize=15, label='User 3')
	# ax.plot(np.array([user3_robot[0, 0]]), np.array([user3_robot[0, 1]]), np.array([user3_robot[0, 2]]),
	#         color='blue', marker='2', markersize=15, label='Robot')
	# ax.plot(np.array([user3_targets[0, 0]]), np.array([user3_targets[0, 1]]), np.array([user3_targets[0, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.plot(np.array([user3_targets[1, 0]]), np.array([user3_targets[1, 1]]), np.array([user3_targets[1, 2]]),
	#         color='darkorange', marker='D', markersize=10)
	# ax.set_xlim3d(-500, 500)
	# ax.set_ylim3d(-2000, 2000)
	# ax.set_zlim3d(-1500, 1500)
	# plt.legend(loc='best')
	# fig4.show()

	# input()

	targets_pos = np.array(targets.values())
	# Trajectory optimization
	optimization_criteria = 'avg'
	# optimization_criteria = 'minmax'
	legible_movement = LegibleMovement(n_targets=n_targets, targets_pos=targets_pos, n_users=n_users, using_ros=False,
	                                   user_poses=user_poses, robot_pose=robot_pose,
	                                   orientation_type='euler', u_ids=u_ids)

	# Run optimization
	optim_traj = traj_1.copy()
	prev_optim_traj = traj_1.copy()
	optim_legibility = 0
	optim_user_legibilities = []
	prev_optim_legibility = 0
	prev_optim_user_legibilities = []
	# n_iterations = 1000
	n_iterations = 100000
	learn_rate = 0.001
	improved_trajs = OrderedDict()
	trajs_legibility = OrderedDict()
	optim_iteration = 0

	for i in range(n_iterations):
		print('Iteration: ' + str(i+1))
		improve_trajs, best_user = legible_movement.improveTrajectory(robot_target=optim_target, trajectory=optim_traj,
		                                                              optimization_criteria=optimization_criteria,
		                                                              learn_rate=learn_rate)
		improved_traj = improve_trajs[best_user]
		impossible_traj = np.isnan(improved_traj).any()
		improved_legibility = 0
		users = legible_movement.get_users()
		transformed_trajs = []
		transformed_legibility = []
		t_targets = {}

		for user in u_ids:
			traj_leg = users[user].trajectoryLegibility(targets_pos, improved_traj, has_transform=True)

			impossible_traj = np.isnan(traj_leg)
			if impossible_traj:
				break

			improved_legibility += traj_leg
			if i == 0 or (i < 1001 and (i + 1) % 100 == 0) or (i + 1) % 1000 == 0:
				print('User: %s\tLegibility: %.5f' % (user, traj_leg))

			t_targets[user] = users[user].transformTrajectory(targets_pos, viewport=False)
			transformed_trajs += [users[user].transformTrajectory(improved_traj, viewport=False)]
			transformed_legibility += [traj_leg]

		if not impossible_traj:

			improved_legibility = improved_legibility/float(len(u_ids))
			if improved_legibility > optim_legibility:
				prev_optim_traj = optim_traj
				prev_optim_legibility = optim_legibility
				prev_optim_user_legibilities = optim_user_legibilities
				optim_traj = improved_traj
				optim_legibility = improved_legibility
				optim_user_legibilities = transformed_legibility
				optim_iteration += 1

				if i == 0 or (i < 1001 and (i + 1) % 100 == 0) or (i+1) % 1000 == 0:
					print('Average Legibility: %.5f' % improved_legibility)
					improved_trajs[str(i + 1)] = improved_traj
					trajs_legibility[str(i + 1)] = [transformed_legibility, improved_legibility]

					# if i == 0:
					#
					# 	fig1 = plt.figure('Improved Trajectory')
					# 	plt.clf()
					# 	ax = fig1.gca(projection='3d')
					# 	ax.plot(np.array([user1_translation[0]]), np.array([user1_translation[1]]),
					# 	        np.array([user1_translation[2]]),
					# 	        color='red', marker='2', markersize=15, label='User1')
					# 	ax.plot(np.array([user2_translation[0]]), np.array([user2_translation[1]]),
					# 	        np.array([user2_translation[2]]),
					# 	        color='green', marker='2', markersize=15, label='User2')
					# 	ax.plot(np.array([user3_translation[0]]), np.array([user3_translation[1]]),
					# 	        np.array([user3_translation[2]]),
					# 	        color='brown', marker='2', markersize=15, label='User3')
					# 	ax.plot(np.array([targets['A'][0]]), np.array([targets['A'][1]]), np.array([targets['A'][2]]),
					# 	        color='darkorange', marker='D', markersize=10)
					# 	ax.plot(np.array([targets['B'][0]]), np.array([targets['B'][1]]), np.array([targets['B'][2]]),
					# 	        color='darkorange', marker='D', markersize=10)
					# 	ax.plot(np.array([targets['C'][0]]), np.array([targets['C'][1]]), np.array([targets['C'][2]]),
					# 	        color='darkorange', marker='D', markersize=10)
					# 	ax.plot(np.array([robot_translation[0]]), np.array([robot_translation[1]]),
					# 	        np.array([robot_translation[2]]),
					# 	        color='blue', marker='o', markersize=10, label='Robot')
					# 	ax.plot(np.array([optim_traj[0, 0]]), np.array([optim_traj[0, 1]]), np.array([optim_traj[0, 2]]),
					# 	        color='black', marker='*', markersize=10, label='Start')
					# 	ax.plot(np.array([optim_traj[-1, 0]]), np.array([optim_traj[-1, 1]]), np.array([optim_traj[-1, 2]]),
					# 	        color='gold', marker='*', markersize=10, label='Goal')
					# 	ax.plot(optim_traj[:, 0], optim_traj[:, 1], optim_traj[:, 2], 'green', markersize=10)
					# 	ax.view_init(azim=-30, elev=20)
					# 	ax.set_xlim3d(-100, 3000)
					# 	ax.set_ylim3d(-100, 3000)
					# 	ax.set_zlim3d(0, 250)
					# 	plt.legend(loc='best')
					# 	plt.draw()
					# 	plt.pause(5.0)

			else:
				break

		else:
			optim_traj = prev_optim_traj
			optim_legibility = prev_optim_legibility
			optim_user_legibilities = prev_optim_user_legibilities

			print('Improved trajectory unfeasable. Stopping optimization process')
			break

	fig1 = plt.figure('Improved Trajectory')
	plt.clf()
	ax = fig1.gca(projection='3d')
	ax.plot(np.array([user1_translation[0]]), np.array([user1_translation[1]]), np.array([user1_translation[2]]),
	        color='red', marker='2', markersize=15, label='User1')
	ax.plot(np.array([user2_translation[0]]), np.array([user2_translation[1]]), np.array([user2_translation[2]]),
	        color='green', marker='2', markersize=15, label='User2')
	ax.plot(np.array([user3_translation[0]]), np.array([user3_translation[1]]), np.array([user3_translation[2]]),
	        color='brown', marker='2', markersize=15, label='User3')
	ax.plot(np.array([targets['A'][0]]), np.array([targets['A'][1]]), np.array([targets['A'][2]]),
	        color='darkorange', marker='D', markersize=10)
	ax.plot(np.array([targets['B'][0]]), np.array([targets['B'][1]]), np.array([targets['B'][2]]),
	        color='darkorange', marker='D', markersize=10)
	ax.plot(np.array([targets['C'][0]]), np.array([targets['C'][1]]), np.array([targets['C'][2]]),
	        color='darkorange', marker='D', markersize=10)
	ax.plot(np.array([robot_translation[0]]), np.array([robot_translation[1]]), np.array([robot_translation[2]]),
	        color='blue', marker='o', markersize=10, label='Robot')
	ax.plot(np.array([optim_traj[0, 0]]), np.array([optim_traj[0, 1]]), np.array([optim_traj[0, 2]]),
	        color='black', marker='*', markersize=10, label='Start')
	ax.plot(np.array([optim_traj[-1, 0]]), np.array([optim_traj[-1, 1]]), np.array([optim_traj[-1, 2]]),
	        color='gold', marker='*', markersize=10, label='Goal')
	ax.plot(optim_traj[:, 0], optim_traj[:, 1], optim_traj[:, 2], 'green', markersize=10)
	ax.set_xlim3d(-100, 3000)
	ax.set_ylim3d(-100, 3000)
	ax.set_zlim3d(0, 250)
	plt.legend(loc='best')
	fig1.show()

	print('Optim Trajctory')
	print(optim_traj)

	print('Optim Trajectory Legibility')
	print(optim_legibility)
	print('Optim User Legibilities')
	for i in range(n_users):
		print('%s legibility: %.5f' % (u_ids[i].capitalize(), optim_user_legibilities[i]))

	print('Difference with Original')
	print((optim_traj - traj_1))
	print('\n')

	# Legibility values for all users - For evaluation purposes
	eval_legibility = LegibleMovement(n_targets=3, targets_pos=targets_pos, n_users=len(user_defined_ids), using_ros=False,
	                                  user_poses=user_poses_defined, robot_pose=robot_pose,
	                                  orientation_type='euler', u_ids=user_defined_ids)
	eval_users = eval_legibility.get_users()
	eval_leg_avg = 0
	optim_legs = []
	print('Legibility for all possible users')
	for user in user_defined_ids:
		eval_users[user].updateTarget(optim_target)
		user_leg = eval_users[user].trajectoryLegibility(targets=targets_pos, orig_trajectory=optim_traj, has_transform=True)
		user_leg = 0 if np.isnan(user_leg) else user_leg
		print('%s legibility: %.5f' % (user.capitalize(), user_leg))
		eval_leg_avg += user_leg
		optim_legs += [user_leg]

	optim_avg_leg = eval_leg_avg/len(user_defined_ids)
	print('Average Legibility: %.5f' % (optim_avg_leg))

	improved_trajs[str(optim_iteration)] = optim_traj
	trajs_legibility[str(optim_iteration)] = [optim_legs, optim_avg_leg]

	# Recording trajectories and legibilities to file
	print('Storing Trajectories to file')
	write_file = open(filename, "w")
	writer = csv.writer(write_file)
	writer.writerow(['Iteration', 'Trajectory', 'Legibility'])
	for key in improved_trajs.keys():
		writer.writerow([key, improved_trajs[key], trajs_legibility[key]])
	write_file.close()

	print('------------------------------------------')
	print('-------- OPTIMIZATION PROGRAM ENDED ------')
	print('------------------------------------------')

	x = raw_input()


if __name__ == '__main__':
	main()

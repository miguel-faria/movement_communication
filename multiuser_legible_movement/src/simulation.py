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


def run_simulation(filename, learn_rate, optimization_criteria, n_iterations, optim_target, traj, n_targets, targets,
				   targets_pos, n_users, user_poses, u_ids, robot_pose, horizontal_fov, vertical_fov,
				   user_defined_ids, user_poses_defined, plotting=False, write_mode='a', store_mode='all'):

	user1_translation = user_poses_defined[0].position
	user2_translation = user_poses_defined[1].position
	user3_translation = user_poses_defined[2].position
	robot_translation = robot_pose.position

	improved_trajs = OrderedDict()
	trajs_legibility = OrderedDict()
	optim_iteration = 0
	optim_traj = traj.copy()
	prev_optim_traj = traj.copy()
	optim_legibility = 0
	optim_user_legibilities = []
	prev_optim_legibility = 0
	prev_optim_user_legibilities = []
	legible_movement = LegibleMovement(n_targets=n_targets, targets_pos=targets_pos, n_users=n_users, using_ros=False,
									   user_poses=user_poses, robot_pose=robot_pose, w_field_of_view=horizontal_fov,
									   h_field_of_view=vertical_fov, orientation_type='euler', u_ids=u_ids)

	# Start optimization process
	for i in range(n_iterations):
		print('Iteration: ' + str(i + 1))
		improve_trajs, best_user = legible_movement.improveTrajectory(robot_target=optim_target, trajectory=optim_traj,
																	  optimization_criteria=optimization_criteria,
																	  learn_rate=learn_rate)
		improved_traj = improve_trajs[best_user]
		impossible_traj = np.isnan(improved_traj).any()
		improved_legibility = 0
		users = legible_movement.get_users()
		transformed_legibility = []

		# Verify if improved trajectory is possible for all users
		for user in u_ids:
			traj_leg = users[user].trajectoryLegibility(targets_pos, improved_traj, has_transform=True)

			impossible_traj = np.isnan(traj_leg)
			if impossible_traj:
				break

			improved_legibility += traj_leg
			transformed_legibility += [traj_leg]
			if i == 0 or (i < 1001 and (i + 1) % 100 == 0) or (i + 1) % 1000 == 0:
				print('User: %s\tLegibility: %.5f' % (user, traj_leg))

		# In case of a possible trajectory update optimization results
		if not impossible_traj:
			improved_legibility = improved_legibility / float(len(u_ids))
			print('Legibility improvement: %.9f' % (improved_legibility - optim_legibility))
			if improved_legibility >= optim_legibility and (abs(improved_legibility - optim_legibility) > 1e-13):
				prev_optim_traj = optim_traj
				prev_optim_legibility = optim_legibility
				prev_optim_user_legibilities = optim_user_legibilities
				optim_traj = improved_traj
				optim_legibility = improved_legibility
				optim_user_legibilities = transformed_legibility
				optim_iteration += 1

				if i == 0 or (i < 1001 and (i + 1) % 100 == 0) or (i + 1) % 1000 == 0:
					print('Average Legibility: %.5f' % improved_legibility)
					improved_trajs[str(i + 1)] = improved_traj
					trajs_legibility[str(i + 1)] = [transformed_legibility, improved_legibility]

			else:
				break

		# In case of an impossible trajectory, break optimization loop
		else:
			optim_traj = prev_optim_traj
			optim_legibility = prev_optim_legibility
			optim_user_legibilities = prev_optim_user_legibilities

			print('Improved trajectory unfeasable. Stopping optimization process')
			break

	# Plot of optimized and original trajectories
	if plotting:
		fig1 = plt.figure('Optimized vs Original Trajectory')
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
		ax.plot(optim_traj[:, 0], optim_traj[:, 1], optim_traj[:, 2], 'green', markersize=10, marker='.',
				label='Optimized Trajectory')
		ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'blue', markersize=10, marker='.', label='Original Trajectory')
		plt.legend(loc='best')
		fig1.show()

	# Output optimization results
	print('----------------------------------------')
	print('|--------------------------------------|')
	print('|                                      |')
	print('|         Optimization Results         |')
	print('|                                      |')
	print('|--------------------------------------|')
	print('----------------------------------------')
	print('Optim Trajctory:')
	print(optim_traj)
	print('------------------------------------')
	print('Optim Trajectory Legibility: %.5f' % optim_legibility)
	print('------------------------------------')
	print('Optim User Legibilities:')
	if n_users > 1:
		for i in range(n_users):
			print('%s legibility: %.5f' % (u_ids[i].capitalize(), optim_user_legibilities[i]))
	print('------------------------------------')
	print('\n')

	# Legibility values for all users - For evaluation purposes
	print('-------------------------------------------------')
	print('|-----------------------------------------------|')
	print('|                                               |')
	print('|         Results For All Defined Users         |')
	print('|                                               |')
	print('|-----------------------------------------------|')
	print('-------------------------------------------------')
	eval_legibility = LegibleMovement(n_targets=3, targets_pos=targets_pos, n_users=len(user_defined_ids),
									  using_ros=False,
									  user_poses=user_poses_defined, robot_pose=robot_pose,
									  w_field_of_view=horizontal_fov,
									  h_field_of_view=vertical_fov, orientation_type='euler', u_ids=user_defined_ids)
	eval_users = eval_legibility.get_users()
	eval_leg_avg = 0
	optim_legs = []
	print('------------------------')
	print('Legibility for each user')
	for user in user_defined_ids:
		eval_users[user].updateTarget(optim_target)
		user_leg = eval_users[user].trajectoryLegibility(targets=targets_pos, orig_trajectory=optim_traj,
														 has_transform=True)
		user_leg = 0 if np.isnan(user_leg) else user_leg
		print('%s legibility: %.5f' % (user.capitalize(), user_leg))
		eval_leg_avg += user_leg
		optim_legs += [user_leg]
	optim_avg_leg = eval_leg_avg / len(user_defined_ids)
	print('-------------------------')
	print('Average Legibility: %.5f' % optim_avg_leg)
	print('-------------------------')
	print('\n\n\n')

	# Storing trajectories and legibilities in file
	improved_trajs[str(optim_iteration)] = optim_traj
	trajs_legibility[str(optim_iteration)] = [optim_legs, optim_avg_leg]
	print('--------------------------------')
	print('| Storing Trajectories to file |')
	print('--------------------------------')
	write_file = open(filename, write_mode)
	writer = csv.writer(write_file)
	writer.writerow(['Iteration', 'Trajectory', 'Legibility'])
	if store_mode.find('all') != -1:
		for key in improved_trajs.keys():
			writer.writerow([key, improved_trajs[key], trajs_legibility[key]])
	elif store_mode.find('optim') != -1:
		key = str(optim_iteration)
		writer.writerow([key, improved_trajs[key], trajs_legibility[key]])
	else:
		print('[SIMULATION ERROR] Invalid storing mode given. No results saved to file.')
	write_file.close()


def main():

	file_path = os.path.dirname(sys.argv[0])
	full_path = os.path.abspath(file_path)
	image_dir = full_path + '/images'

	# Configuration 1
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	# Configuration 2
	user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	user2_rot = T.quaternion_from_euler(ai=math.radians(300), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	user3_rot = T.quaternion_from_euler(ai=math.radians(60), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# Configuration 3
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(70), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(-30), aj=math.radians(70), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(30), aj=math.radians(70), ak=math.radians(0), axes='rzxy')
	# Configuration 4
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(-50), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(50), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# Configuration 5
	# user1_rot = T.quaternion_from_euler(ai=math.radians(315), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(295), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# Configuration 6
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(330), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(315), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# Configuration 7
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(-80), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(80), aj=math.radians(75), ak=math.radians(0), axes='rzxy')
	# Configuration 8
	# user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(65), ak=math.radians(0), axes='rzxy')
	# Robot rotation
	robot_rot = T.quaternion_from_euler(ai=math.radians(180), aj=math.radians(70), ak=math.radians(0), axes='rzxy')

	# User Orientation Simulation
	# user1_rot = T.quaternion_from_euler(ai=math.radians(180), aj=math.radians(-10), ak=math.radians(0), axes='ryxz')
	# user2_rot = T.quaternion_from_euler(ai=math.radians(90), aj=math.radians(-10), ak=math.radians(0), axes='ryxz')
	# user3_rot = T.quaternion_from_euler(ai=math.radians(270), aj=math.radians(-10), ak=math.radians(0), axes='ryxz')
	# Robot Orientarion Simulation
	# robot_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(0), ak=math.radians(0), axes='ryxz')

	# Configuration 1-4, 7-8
	user1_translation = (1500.0, 1000.0, 1000.0)
	user2_translation = (500.0, 1500.0, 1000.0)
	user3_translation = (2500.0, 1500.0, 1000.0)
	# Configuration 5
	# user1_translation = (500.0, 1000.0, 1700.0)
	# user2_translation = (500.0, 1500.0, 1700.0)
	# user3_translation = (500.0, 2000.0, 1700.0)
	# Configuration 6
	# user1_translation = (1500.0, 1000.0, 1700.0)
	# user2_translation = (1000.0, 1200.0, 1700.0)
	# user3_translation = (500.0, 1400.0, 1700.0)
	# Robot Position
	robot_translation = (1500.0, 2500.0, 250.0)

	# User Position Simulation
	# user1_translation = (1500.0, 1000.0, 500.0)
	# user2_translation = (3200.0, 1000.0, 1750.0)
	# user3_translation = (-250.0, 1000.0, 1750.0)
	# Robot Position Simulation
	# robot_translation = (1500.0, 250.0, 2400.0)

	# Poses
	robot_pose = Pose(position=robot_translation, orientation=robot_rot)
	user1_pose = Pose(position=user1_translation, orientation=user1_rot)
	user2_pose = Pose(position=user2_translation, orientation=user2_rot)
	user3_pose = Pose(position=user3_translation, orientation=user3_rot)

	# Configuration 1-3
	# targets = {'A': np.array([1200.0, 1300.0, 250.0]), 'B': np.array([1500.0, 1300.0, 250.0]),
	#            'C': np.array([1800.0, 1300.0, 250.0])}
	targets = {'A': np.array([1200.0, 1500.0, 250.0]), 'B': np.array([1500.0, 1500.0, 250.0]),
	           'C': np.array([1800.0, 1500.0, 250.0])}
	# targets = {'A': np.array([1200.0, 2050.0, 250.0]), 'B': np.array([1500.0, 2050.0, 250.0]),
	#            'C': np.array([1800.0, 2050.0, 250.0])}
	# Configuration 4
	# targets = {'A': np.array([1250.0, 1600.0, 250.0]), 'B': np.array([1750.0, 1600.0, 250.0]),
	#            'C': np.array([1500.0, 1300.0, 250.0])}
	# Configuration 5
	# targets = {'A': np.array([1200.0, 1500.0, 250.0]), 'B': np.array([1500.0, 1500.0, 250.0]),
	#            'C': np.array([1800.0, 1500.0, 250.0])}
	# Configuration 6
	# targets = {'A': np.array([1250.0, 1700.0, 250.0]), 'B': np.array([1500.0, 1500.0, 250.0]),
	#            'C': np.array([1800.0, 1300.0, 250.0])}
	# Configuration 7
	# targets = {'A': np.array([1250.0, 1600.0, 250.0]), 'B': np.array([1500.0, 1800.0, 250.0]),
	#            'C': np.array([1750.0, 1600.0, 250.0])}
	# Configuration 8
	# targets = {'A': np.array([1500.0, 1200.0, 250.0]), 'B': np.array([1500.0, 1800.0, 250.0]),
	#            'C': np.array([1500.0, 1500.0, 250.0])}

	# Taeget Position Simulation
	# targets = {'A': np.array([1800.0, 250.0, 1700.0]), 'B': np.array([1500.0, 250.0, 1700.0]),
	#            'C': np.array([1200.0, 250.0, 1700.0])}
	# targets = {'A': np.array([1650.0, 250.0, 1700.0]), 'B': np.array([1500.0, 250.0, 1500.0]),
	# 		   'C': np.array([1300.0, 250.0, 1700.0])}
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
	optimization_criteria = 'avg'
	# optimization_criteria = 'minmax'
	vertical_fov = 55
	horizontal_fov = 120
	n_iterations = 500000
	learn_rate = (1.0 / 0.05)
	n_users = 3
	n_targets = len(targets)
	optim_target = 'A'
	user_poses = [user1_pose, user2_pose, user3_pose]
	# user_poses = [user2_pose]
	u_ids = ['user1', 'user2', 'user3']
	# u_ids = ['user2']
	filename = "../data/3_users_multiple_trajs.csv"
	# filename = "../data/1_user_conf_7_u2_c.csv"
	targets_pos = np.array(targets.values())

	# Variables and definitions for performance evaluation
	user_poses_defined = [user1_pose, user2_pose, user3_pose]
	user_defined_ids = ['user1', 'user2', 'user3']

	# Trajectory Creation
	n_points = 20
	traj_x = np.linspace(robot_translation[0], targets[optim_target][0], num=n_points)[:, None]
	traj_y = np.linspace(robot_translation[1], targets[optim_target][1], num=n_points)[:, None]
	traj_z = np.linspace(targets[optim_target][2], targets[optim_target][2], num=n_points)[:, None]
	offset_modulator = 300 * np.sin(np.linspace(0, np.pi, num=n_points))[:, None]
	# Trajectory Creation Simulation
	# traj_x = np.linspace(robot_translation[0], targets[optim_target][0], num=n_points)[:, None]
	# traj_y = np.linspace(targets[optim_target][1], targets[optim_target][1], num=n_points)[:, None]
	# traj_z = np.linspace(robot_translation[2], targets[optim_target][2], num=n_points)[:, None]

	base_traj = np.hstack((traj_x, traj_y, traj_z))
	h_mod_traj = np.hstack((traj_x, traj_y, traj_z + offset_modulator))
	r_mod_traj = np.hstack((traj_x - offset_modulator, traj_y, traj_z))
	l_mod_traj = np.hstack((traj_x + offset_modulator, traj_y, traj_z))
	hr_mod_traj = np.hstack((traj_x - offset_modulator, traj_y, traj_z + offset_modulator))
	hl_mod_traj = np.hstack((traj_x + offset_modulator, traj_y, traj_z + offset_modulator))
	r_spring_mod_traj = np.hstack((traj_x - offset_modulator, traj_y - offset_modulator, traj_z))
	l_spring_mod_traj = np.hstack((traj_x + offset_modulator, traj_y - offset_modulator, traj_z))
	trajectories_sequence = [base_traj, h_mod_traj, r_mod_traj, l_mod_traj, hr_mod_traj, hl_mod_traj,
					 r_spring_mod_traj, l_spring_mod_traj]

	# user1 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user1',
	#                                   user_pose=user1_pose, robot_pose=robot_pose)
	# user2 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user2',
	#                                   user_pose=user2_pose, robot_pose=robot_pose)
	# user3 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user3',
	#                                   user_pose=user3_pose, robot_pose=robot_pose)

	# user1_traj = user1.transformTrajectory(base_traj, viewport=False)
	# user2_traj = user2.transformTrajectory(base_traj, viewport=False)
	# user3_traj = user3.transformTrajectory(base_traj, viewport=False)

	# print('Original Trajectory')
	# print(base_traj)
	# print('User 1 transformed trajectory')
	# print(user1_traj)
	# print('User 2 transformed trajectory')
	# print(user2_traj)
	# print('User 3 transformed trajectory')
	# print(user3_traj)

	# fig = plt.figure('Original Trajectory')
	# ax = fig.add_subplot(111, projection=Axes3D.name)
	# ax.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], 'black', label='Trajectory', marker='.', linestyle="None")
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

	# fig = plt.figure('Trajectories Visual')
	# ax = fig.add_subplot(111, projection=Axes3D.name)
	# ax.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], 'black', label='Base', marker='.',
	# 		linestyle="None")
	# ax.plot(h_mod_traj[:, 0], h_mod_traj[:, 1], h_mod_traj[:, 2], 'blue', label='Height', marker='.',
	# 		linestyle="None")
	# ax.plot(r_mod_traj[:, 0], r_mod_traj[:, 1], r_mod_traj[:, 2], 'green', label='Right', marker='.',
	# 		linestyle="None")
	# ax.plot(l_mod_traj[:, 0], l_mod_traj[:, 1], l_mod_traj[:, 2], 'cyan', label='Left', marker='.',
	# 		linestyle="None")
	# ax.plot(hr_mod_traj[:, 0], hr_mod_traj[:, 1], hr_mod_traj[:, 2], 'red', label='Right Height', marker='.',
	# 		linestyle="None")
	# ax.plot(hl_mod_traj[:, 0], hl_mod_traj[:, 1], hl_mod_traj[:, 2], 'orange', label='Left Height', marker='.',
	# 		linestyle="None")
	# ax.plot(r_spring_mod_traj[:, 0], r_spring_mod_traj[:, 1], r_spring_mod_traj[:, 2], 'brown', label='Right Spring',
	# 		marker='.', linestyle="None")
	# ax.plot(l_spring_mod_traj[:, 0], l_spring_mod_traj[:, 1], l_spring_mod_traj[:, 2], 'pink', label='Left Spring',
	# 		marker='.', linestyle="None")
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
	# fig.show()
	#
	# x = raw_input()

	# Run simulation
	n_trajectories = 8
	for i in range(n_trajectories):
		traj = trajectories_sequence[i]
		run_simulation(filename, learn_rate, optimization_criteria, n_iterations, optim_target, traj, n_targets, targets,
					   targets_pos, n_users, user_poses, u_ids, robot_pose, horizontal_fov, vertical_fov, user_defined_ids,
					   user_poses_defined, store_mode='optim')

	print('------------------------------------------')
	print('-------- OPTIMIZATION PROGRAM ENDED ------')
	print('------------------------------------------')

	x = raw_input()


if __name__ == '__main__':
	main()

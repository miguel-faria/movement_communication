# !/usr/bin/env python
import numpy as np
import rospy
import string
import roboticstoolbox as rtb
import tf.transformations as T

from geometry_msgs.msg import Quaternion, Point, Pose

from user_perspective_legibility import UserPerspectiveLegibility
from utilites import get_dict_keys, forward_kinematics, cartesian_to_joint_optimization


class LegibleMovement(object):

	def __init__(self, n_targets: int, targets_pos: list, n_users: int, using_ros: bool, user_poses: list,
				 robot_pose: Pose, orientation_type: string, joint_optim: bool, robot_model: rtb.DHRobot = None,
				 u_ids=None, targets_prob=None, w_field_of_view=124, h_field_of_view=60, clip_planes=None):

		if u_ids is None:
			self._u_ids = ['user_' + str(x+1) for x in range(n_users)]
		elif len(u_ids) != n_users:
			print('[LEGIBLE MOVEMENT ERROR] Mismatch in the number of users and corresponding identifications, storing '
			      'identifications until given number.')
			self._u_ids = [u_ids[x] for x in range(n_users)]
		else:
			self._u_ids = u_ids

		if len(user_poses) < n_users:
			print('[LEGIBLE MOVEMENT ERROR] Poses undefined for all users.')
			return

		if targets_prob is None:
			self._targets_prob = {}
			for i in range(n_targets):
				self._targets_prob[string.ascii_uppercase[i]] = 1/float(n_targets)
		else:
			self._targets_prob = targets_prob

		if clip_planes is None:
			clip_planes = [0.9, 5]

		self._joint_optim = joint_optim
		self._robot_model = robot_model

		self._n_targets = n_targets
		self._targets = get_dict_keys(self._targets_prob)
		self._targets_pos = {}
		for i in range(len(self._targets)):
			self._targets_pos[self._targets[i]] = targets_pos[i]

		self._n_users = n_users
		self._user_posers = user_poses
		self._orientation_type = orientation_type
		self._w_field_of_view = w_field_of_view
		self._h_field_of_view = h_field_of_view

		self._users = {}
		for i in range(n_users):
			self._users[self._u_ids[i]] = UserPerspectiveLegibility(using_ros=using_ros, user_pose=user_poses[i],
			                                                        robot_pose=robot_pose, orientation_type=orientation_type,
			                                                        user_id=self._u_ids[i], targets_prob=self._targets_prob,
			                                                        w_fov=w_field_of_view, h_fov=h_field_of_view,
			                                                        clip_planes=clip_planes)
		self._robot_pose = robot_pose

	def get_user_ids(self):
		return self._u_ids

	def get_users(self):
		return self._users

	def get_user_posers(self):
		return self._user_posers

	def gradientStep(self, trajectory: np.ndarray, robot_target: str, trajectory_costs: dict,
					 gradient_costs: dict, user: str):

		traj_len = len(trajectory)

		user_target_best_costs = trajectory_costs[user][robot_target][1]
		user_target_grad_costs = gradient_costs[user][robot_target]
		cost_goal = np.exp(user_target_best_costs[0] - user_target_best_costs[:])
		cost_goals = (np.exp(user_target_best_costs[0] - user_target_best_costs[:]) *
		              self._targets_prob[robot_target])
		cost_grad = ((np.exp(-user_target_best_costs[:]) * self._targets_prob[robot_target] /
		              np.exp(-user_target_best_costs[0]))[:, None] * (user_target_grad_costs - user_target_grad_costs)) * 1e4

		for target in self._targets:

			if target.find(robot_target) == -1:
				user_target_best_costs = trajectory_costs[user][target][1]
				user_target_grad_costs = gradient_costs[user][target]
				cost_goals += (np.exp(user_target_best_costs[0] - user_target_best_costs[:]) *
				               self._targets_prob[target])
				cost_grad += ((np.exp(-user_target_best_costs[:]) * self._targets_prob[target] /
				               np.exp(-user_target_best_costs[0]))[:, None] *
				              (user_target_grad_costs[:] - gradient_costs[user][robot_target][:])) * 1e4

		time_function = np.array([(traj_len - i) / float(traj_len) for i in range(traj_len)])

		legibility_grad = (cost_goal / cost_goals**2)[:, None] * self._targets_prob[robot_target] * cost_grad * time_function[:, None]

		return legibility_grad / np.sum(time_function)

	def max_optimization(self, trajectory: np.ndarray, legibility_grads, learn_rate: float = 0.001):

		optim_trajs = {}
		traj_len, traj_dim = trajectory.shape
		grad_sum = np.zeros((self._n_users, traj_len, traj_dim))

		for i in range(self._n_users):

			user = self._u_ids[i]
			user_M, _ = self._users[user].getCostMatrices(trajectory)
			projection_grad = self._users[user].trajectory2DProjectGrad(trajectory)
			user_M = user_M.T.dot(user_M)
			M_inv = np.linalg.inv(user_M)

			gradient = []
			for j in range(traj_len):
				gradient += [legibility_grads[user][j, :].dot(projection_grad[j])]

			gradient = np.array(gradient)

			for j in range(traj_dim):
				grad_sum[i, :, j] += M_inv.dot(gradient[:, j])

		best_grad = np.argmax(np.sum(grad_sum, axis=0))

		grad_step = float(1 / learn_rate) * grad_sum[best_grad]
		grad_step[0] = 0
		grad_step[-1] = 0
		improve_traj = trajectory.copy()
		improve_traj += grad_step

		optim_trajs[self._u_ids[0]] = improve_traj
		best_traj_user = self._u_ids[0]

		return optim_trajs, best_traj_user

	def user_average_optimization(self, world_trajectory, legibility_grads, joint_trajectory, learn_rate=0.001):

		optim_trajs = {}
		traj_len, traj_dim = world_trajectory.shape
		grad_sum = np.zeros((traj_len, traj_dim))

		for i in range(self._n_users):

			user = self._u_ids[i]
			user_M, _ = self._users[user].getCostMatrices(world_trajectory)
			transform_matrix, _ = self._users[user].getTransformationMatrices()
			transform_matrix = np.delete(np.delete(transform_matrix, -1, 0), -1, 1)
			projection_grad = self._users[user].trajectory2DProjectGrad(world_trajectory)
			user_M = user_M.T.dot(user_M)
			M_inv = np.linalg.inv(user_M)

			gradient = []
			for j in range(traj_len):
				gradient += [legibility_grads[user][j, :].dot(projection_grad[j])]

			gradient = np.array(gradient)
			gradient = gradient.dot(transform_matrix)

			for j in range(traj_dim):
				grad_sum[:, j] += M_inv.dot(gradient[:, j])

		grad_step = (grad_sum / float(self._n_users))
		grad_step[0] = 0
		grad_step[-1] = 0

		if self._joint_optim:
			improve_traj = joint_trajectory.copy()
			grad_step *= 1/1000
			joint_grad = cartesian_to_joint_optimization(self._robot_model, grad_step, joint_trajectory)
			joint_limits = self._robot_model.qlim
			improve_traj += float(1/learn_rate) * joint_grad
			for i in range(improve_traj.shape[1]):
				improve_traj[:, i] = np.maximum(np.minimum(improve_traj[:, i], joint_limits[1, i]), joint_limits[0, i])
		else:
			improve_traj = world_trajectory.copy()
			improve_traj += float(1/learn_rate) * grad_step

		optim_trajs[self._u_ids[0]] = improve_traj
		best_traj_user = self._u_ids[0]

		return optim_trajs, best_traj_user

	def get_robot_transformation(self):

		robot_orientation = Quaternion(self._robot_pose.orientation[0], self._robot_pose.orientation[1],
									   self._robot_pose.orientation[2], self._robot_pose.orientation[3])
		robot_position = Point(self._robot_pose.position[0], self._robot_pose.position[1], self._robot_pose.position[2])

		robot_euler = T.euler_from_quaternion((robot_orientation.x, robot_orientation.y,
												robot_orientation.z, robot_orientation.w))

		robot_transformation = T.euler_matrix(robot_euler[0], robot_euler[1], robot_euler[2])

		robot_transformation[0, 3] = robot_position.x
		robot_transformation[1, 3] = robot_position.y
		robot_transformation[2, 3] = robot_position.z

		return np.linalg.inv(robot_transformation)

	def improveTrajectory(self, robot_target, trajectory, optimization_criteria, learn_rate):

		traj_costs = {}
		grad_costs = {}
		grad_legibility = {}

		# convert from joint space to world space if need be
		if self._joint_optim:
			fk_trajectory = forward_kinematics(self._robot_model, trajectory)		# get trajectory in robot perspective
			transformation_matrix = np.linalg.inv(self.get_robot_transformation())	# get robot2world transformation

			world_trajectory = []
			for i in range(len(fk_trajectory)):
				world_trajectory += [transformation_matrix.dot(np.concatenate((fk_trajectory[i], 1), axis=None))[:-1]]
			world_trajectory = np.array(world_trajectory)
		else:
			world_trajectory = trajectory

		for user_id in self._u_ids:

			self._users[user_id].updateTarget(robot_target)
			costs, best_costs = self._users[user_id].trajectoryCosts(orig_trajectory=world_trajectory,
																	 has_transform=True)
			grad_cost = self._users[user_id].trajectoryGradCost(orig_trajectory=world_trajectory)
			traj_costs[user_id] = {}
			grad_costs[user_id] = {}
			traj_costs[user_id][robot_target] = np.vstack((costs, best_costs))
			grad_costs[user_id][robot_target] = grad_cost

			for target in self._targets:

				if target.find(robot_target) == -1:

					tmp_trajectory = world_trajectory.copy()
					tmp_trajectory[-1] = self._targets_pos[target]
					costs, best_costs = self._users[user_id].trajectoryCosts(orig_trajectory=tmp_trajectory,
					                                                         has_transform=True)
					grad_cost = self._users[user_id].trajectoryGradCost(orig_trajectory=tmp_trajectory)
					traj_costs[user_id][target] = np.vstack((costs, best_costs))
					grad_costs[user_id][target] = grad_cost

			grad_legibility[user_id] = self.gradientStep(world_trajectory, robot_target, traj_costs, grad_costs,
														 user_id)

		if optimization_criteria.find('average') != -1 or optimization_criteria.find('avg') != -1:
			improve_trajs, best_traj_user = self.user_average_optimization(world_trajectory, grad_legibility,
																		   trajectory, learn_rate)

		else:
			improve_trajs, best_traj_user = self.max_optimization(world_trajectory, grad_legibility, learn_rate)

		return improve_trajs, best_traj_user

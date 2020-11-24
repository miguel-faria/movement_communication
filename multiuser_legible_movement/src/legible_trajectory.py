# !/usr/bin/env python
import numpy as np
import rospy
import string
import roboticstoolbox as rtb
import tf.transformations as T

from geometry_msgs.msg import Quaternion, Point, Pose

from user_perspective_legibility import UserPerspectiveLegibility
from utilites import get_dict_keys, get_forward_kinematics, cartesian_to_joint_optimization, transform_world


class LegibleMovement(object):

	def __init__(self, n_targets: int, targets_pos: list, n_users: int, using_ros: bool, user_poses: list,
				 robot_pose: Pose, orientation_type: string, regularizaiton: float, joint_optim: bool,
				 robot_model: rtb.DHRobot = None, model_pose: np.ndarray = np.eye(4), u_ids=None, targets_prob=None,
				 w_field_of_view=124, h_field_of_view=60, clip_planes=None):

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
		self._model_pose = model_pose
		self._regularization = regularizaiton

	def get_user_ids(self):
		return self._u_ids

	def get_users(self):
		return self._users

	def get_user_posers(self):
		return self._user_posers

	def gradientStep(self, trajectory: np.ndarray, robot_target: str, trajectory_costs: dict,
					 gradient_costs: dict, user: str):

		traj_len = len(trajectory)

		user_target_best_costs = trajectory_costs[user][robot_target]
		user_target_grad_costs = gradient_costs[user][robot_target]
		cost_goal = np.exp(user_target_best_costs[0] - user_target_best_costs[:])
		cost_goals = (np.exp(user_target_best_costs[0] - user_target_best_costs[:]) *
		              self._targets_prob[robot_target])
		cost_grad = ((np.exp(-user_target_best_costs[:]) * self._targets_prob[robot_target] /
		              np.exp(-user_target_best_costs[0]))[:, None] * (user_target_grad_costs - user_target_grad_costs)) * 1e4

		for target in self._targets:

			if target.find(robot_target) == -1:
				user_target_best_costs = trajectory_costs[user][target]
				user_target_grad_costs = gradient_costs[user][target]
				cost_goals += (np.exp(user_target_best_costs[0] - user_target_best_costs[:]) *
				               self._targets_prob[target])
				cost_grad += ((np.exp(-user_target_best_costs[:]) * self._targets_prob[target] /
				               np.exp(-user_target_best_costs[0]))[:, None] *
				              (user_target_grad_costs[:] - gradient_costs[user][robot_target][:])) * 1e4

		time_function = np.array([(traj_len - i) / float(traj_len) for i in range(traj_len)])

		target_prob = self._targets_prob[robot_target]
		legibility_grad = (cost_goal / cost_goals**2)[:, None] * target_prob * cost_grad * time_function[:, None]

		return legibility_grad / np.sum(time_function)

	def gradientRegularization(self, world_trajectory: np.ndarray, legibility_gradient: np.ndarray,
							   trajectory_gradient: np.ndarray, user_id: string):

		regularization = np.zeros(world_trajectory.shape)
		user_M, _ = self._users[user_id].getCostMatrices(world_trajectory)
		user_M = user_M.T.dot(user_M)
		M_inv = np.linalg.inv(user_M)

		for i in range(world_trajectory.shape[1] - 1):
			i_traj_grad = trajectory_gradient[:, i][:, None]
			i_leg_grad = legibility_gradient[:, i][:, None]
			aux = np.linalg.inv(i_traj_grad.T.dot(M_inv).dot(i_traj_grad))
			regularization[:, i] = M_inv.dot(i_traj_grad).dot(aux).dot(i_traj_grad.T).dot(M_inv).dot(i_leg_grad).T

		return regularization / 10000

	def betaRegularization(self, world_trajectory: np.ndarray, trajectory_gradient: np.ndarray,
						   trajectory_costs: np.ndarray, user_id: string):

		regularization = np.zeros(world_trajectory.shape)
		user_M, _ = self._users[user_id].getCostMatrices(world_trajectory)
		user_M = user_M.T.dot(user_M)
		M_inv = np.linalg.inv(user_M)

		beta_cost = trajectory_costs - self._regularization
		for i in range(world_trajectory.shape[1] - 1):
			i_traj_grad = trajectory_gradient[:, i][:, None]
			aux = np.linalg.inv(i_traj_grad.T.dot(M_inv).dot(i_traj_grad))
			regularization[:, i] = M_inv.dot(i_traj_grad).dot(aux).T.dot(np.diag(beta_cost))

		return regularization / 10000

	def update_trajectory(self, grad_step, joint_trajectory, world_trajectory):

		if self._joint_optim:
			improve_traj = joint_trajectory.copy()

			# convert gradient to robot space
			robot_transformation = self.get_robot_transformation()
			robot_transformation = np.delete(np.delete(robot_transformation, -1, 0), -1, 1)
			grad_step_robot = np.concatenate((grad_step.dot(robot_transformation), np.ones((len(grad_step), 1))),
											 axis=1)
			grad_step_robot = grad_step_robot.dot(np.linalg.inv(self._model_pose))[:, :-1]

			# convert from cartesian to joint space
			grad_step_robot /= 1000  # convert form millimeters to meters
			joint_grad = cartesian_to_joint_optimization(self._robot_model, grad_step_robot, joint_trajectory)
			improve_traj += joint_grad

		else:
			improve_traj = world_trajectory.copy()
			improve_traj += grad_step

		return improve_traj

	def max_optimization(self, world_trajectory: np.ndarray, legibility_grads: dict,
						 gradient_regularizations: dict, beta_regularizations: dict,
						 joint_trajectory: np.ndarray, learn_rate: float = 0.01):

		traj_len, traj_dim = world_trajectory.shape
		grad_sum = np.zeros((self._n_users, traj_len, traj_dim))
		grad_reg_sum = np.zeros((self._n_users, traj_len, traj_dim))
		beta_reg_sum = np.zeros((self._n_users, traj_len, traj_dim))

		for i in range(self._n_users):

			user = self._u_ids[i]
			user_M, _ = self._users[user].getCostMatrices(world_trajectory)
			transform_matrix, _ = self._users[user].getTransformationMatrices()
			transform_matrix = np.delete(np.delete(transform_matrix, -1, 0), -1, 1)
			projection_grad = self._users[user].trajectory2DProjectGrad(world_trajectory)
			user_M = user_M.T.dot(user_M)
			M_inv = np.linalg.inv(user_M)

			gradient = []
			gradient_reg = []
			beta_reg = []
			for j in range(traj_len):
				gradient += [legibility_grads[user][j, :].dot(projection_grad[j])]
				gradient_reg += [gradient_regularizations[user][j, :].dot(projection_grad[j])]
				beta_reg += [beta_regularizations[user][j, :].dot(projection_grad[j])]

			gradient = np.array(gradient)
			gradient = gradient.dot(transform_matrix)

			for j in range(traj_dim):
				grad_sum[i, :, j] += M_inv.dot(gradient[:, j])

			gradient_reg = np.array(gradient_reg)
			beta_reg = np.array(beta_reg)
			grad_reg_sum[i] += gradient_reg.dot(transform_matrix)
			beta_reg_sum[i] += beta_reg.dot(transform_matrix)

		best_grad = np.argmax(np.sum(grad_sum, axis=0))

		learn_rate = float(learn_rate)
		grad_step = learn_rate * grad_sum[best_grad] - learn_rate * grad_reg_sum[best_grad] - beta_reg_sum[best_grad]
		grad_step[0] = 0
		grad_step[-1] = 0

		return self.update_trajectory(grad_step, joint_trajectory, world_trajectory)

	def user_average_optimization(self, world_trajectory: np.ndarray, legibility_grads: dict,
								  gradient_regularizations: dict, beta_regularizations: dict,
								  joint_trajectory: np.ndarray, learn_rate: float = 0.01):

		traj_len, traj_dim = world_trajectory.shape
		grad_sum = np.zeros((traj_len, traj_dim))
		grad_reg_sum = np.zeros((traj_len, traj_dim))
		beta_reg_sum = np.zeros((traj_len, traj_dim))

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

			grad_reg_sum += gradient_regularizations[user].dot(transform_matrix)
			beta_reg_sum += beta_regularizations[user].dot(transform_matrix)

		grad_sum = grad_sum / float(self._n_users)
		grad_reg_sum = grad_reg_sum / float(self._n_users)
		beta_reg_sum = beta_reg_sum / float(self._n_users)
		learn_rate = float(learn_rate)
		grad_step = learn_rate * grad_sum - learn_rate * grad_reg_sum - beta_reg_sum
		grad_step[0] = 0
		grad_step[-1] = 0

		return self.update_trajectory(grad_step, joint_trajectory, world_trajectory)

	def user_max_min_optimization(self, world_trajectory: np.ndarray, legibility_grads: dict,
								  gradient_regularizations: dict, beta_regularizations: dict,
								  joint_trajectory: np.ndarray, learn_rate: float=0.01):

		updated_trajs = {}
		updated_legs = {}
		traj_len, traj_dim = world_trajectory.shape

		for user in self._u_ids:

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
				gradient[:, j] = M_inv.dot(gradient[:, j])

			gradient_reg = gradient_regularizations[user].dot(transform_matrix)
			beta_reg = beta_regularizations[user].dot(transform_matrix)

			learn_rate = float(learn_rate)
			grad_step = learn_rate * gradient - learn_rate * gradient_reg - beta_reg
			grad_step[0] = 0
			grad_step[-1] = 0

			user_traj = self.update_trajectory(grad_step, joint_trajectory, world_trajectory)
			tmp_legs = np.array([])
			if self._joint_optim:
				user_world_traj = transform_world(
					get_forward_kinematics(self._robot_model)(self._robot_model, user_traj), self._robot_pose)
			else:
				user_world_traj = user_traj

			for user_2 in self._u_ids:
				tmp_legs = np.concatenate((tmp_legs,
										   np.array([self._users[user_2].trajectoryLegibility(
											   list(self._targets_pos.values()),
											   user_world_traj,
											   has_transform=True)])))

			updated_legs[user] = np.min(tmp_legs)
			updated_trajs[user] = user_traj

		best_user = list(updated_trajs.keys())[list(updated_legs.values()).index(max(list(updated_legs.values())))]
		return updated_trajs[best_user]

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
		future_costs = {}
		prev_traj_grad_costs = {}
		remain_traj_grad_costs = {}
		grad_legibility = {}
		grad_legibility_reg = {}
		beta_reg = {}

		# convert from joint space to world space if need be
		if self._joint_optim:
			fk_trajectory = get_forward_kinematics(self._robot_model)(self._robot_model, trajectory)
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
			prev_traj_grad_costs[user_id] = self._users[user_id].trajectoryGradCost(orig_trajectory=world_trajectory)
			remain_traj_grad = self._users[user_id].trajectoryRemainGradCost(orig_trajectory=world_trajectory)
			traj_costs[user_id] = {}
			future_costs[user_id] = {}
			remain_traj_grad_costs[user_id] = {}
			traj_costs[user_id][robot_target] = costs
			future_costs[user_id][robot_target] = best_costs
			remain_traj_grad_costs[user_id][robot_target] = remain_traj_grad

			for target in self._targets:

				if target.find(robot_target) == -1:

					tmp_trajectory = world_trajectory.copy()
					tmp_trajectory[-1] = self._targets_pos[target]
					costs, best_costs = self._users[user_id].trajectoryCosts(orig_trajectory=tmp_trajectory,
					                                                         has_transform=True)
					remain_traj_grad = self._users[user_id].trajectoryRemainGradCost(orig_trajectory=tmp_trajectory)
					traj_costs[user_id][target] = costs
					future_costs[user_id][target] = best_costs
					remain_traj_grad_costs[user_id][target] = remain_traj_grad

			grad_legibility[user_id] = self.gradientStep(world_trajectory, robot_target, future_costs,
														 remain_traj_grad_costs, user_id)
			if self._regularization > 0:
				grad_legibility_reg[user_id] = self.gradientRegularization(world_trajectory, grad_legibility[user_id],
																		   prev_traj_grad_costs[user_id], user_id)
				beta_reg[user_id] = self.betaRegularization(world_trajectory, prev_traj_grad_costs[user_id],
															traj_costs[user_id][robot_target], user_id)

			else:
				grad_legibility_reg[user_id] = np.zeros(world_trajectory.shape)
				beta_reg[user_id] = np.zeros(world_trajectory.shape)

		if optimization_criteria.find('average') != -1 or optimization_criteria.find('avg') != -1:
			improved_traj = self.user_average_optimization(world_trajectory, grad_legibility, grad_legibility_reg,
														   beta_reg, trajectory, learn_rate)
		elif optimization_criteria.find('min') != -1 and optimization_criteria.find('max') != -1:
			improved_traj = self.user_max_min_optimization(world_trajectory, grad_legibility, grad_legibility_reg,
														   beta_reg, trajectory, learn_rate)
		else:
			improved_traj = self.max_optimization(world_trajectory, grad_legibility, grad_legibility_reg,
														   beta_reg, learn_rate)

		return improved_traj

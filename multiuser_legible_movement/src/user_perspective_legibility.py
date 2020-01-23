# !/usr/bin/env python
import numpy as np
import rosgraph
import tf
import rospy
import tf.transformations as T
import math

from geometry_msgs.msg import Point, PointStamped, Quaternion, Pose
#from autograd import grad
#from sympy import principal_branch

np.set_printoptions(precision=9, linewidth=2000, threshold=10000, suppress=True)


class UserPerspectiveLegibility(object):

	def __init__(self, using_ros, user_pose, robot_pose, orientation_type, user_id,
				 target=None, targets_prob=None, w_fov=124, h_fov=60, clip_planes=None):

		if targets_prob is None:
			self._targets_prob = {'A': 0.5, 'B': 0.5}
		else:
			self._targets_prob = targets_prob

		if clip_planes is None:
			clip_planes = [1, 1000]

		self._target = target
		self._user_angle = 0.0
		self._user_pose = user_pose
		self._rotation_type = orientation_type
		self._using_ros = using_ros
		self._user_id = user_id
		w_scale_factor = 1.0/(math.tan((w_fov/2.0) * (math.pi/180)))
		h_scale_factor = 1.0 / (math.tan((h_fov / 2.0) * (math.pi / 180)))
		far_plane = clip_planes[1]
		near_plane = clip_planes[0]
		self._w_fov = w_fov
		self._h_fov = h_fov
		self._clip_planes = clip_planes
		self._perspective_matrix = np.array([[w_scale_factor, 0, 0, 0],
											 [0, h_scale_factor, 0, 0],
											 [0, 0, -float(far_plane)/float(far_plane - near_plane), -1],
											 [0, 0, -float(far_plane*near_plane)/float(far_plane - near_plane), 0]])
		self._robot_pose = robot_pose

		if rosgraph.is_master_online():
			self._ros_active = True
			if self._using_ros:
				self._tf_listener = tf.TransformListener()
			else:
				self._tf_listener = None
		else:
			self._ros_active = False

		self._movement_target = Point()
		self._trajectory = None
		self._transformed_trajectory = None
		self._trajectory_length = -1
		self._trajectory_dim = 0
		self._transformed_trajectory_length = -1
		self._transformed_trajectory_dim = 0
		self._traj_K = None
		self._traj_e = None

	#########################################################################
	####                                                                #####
	####                       GETTERS AND SETTERS                      #####
	####                                                                #####
	#########################################################################

	def updatePos(self, user_pos):
		self._user_pose.position = user_pos

	def updateRotation(self, user_orientation):
		self._user_pose.orientation = user_orientation

	def updatePose(self, user_pose):
		self._user_pose = user_pose

	def updateTarget(self, target):
		if target in self._targets_prob.keys():
			self._target = target
		else:
			if self._using_ros and self._ros_active:
				rospy.logerr('[PERSPECTIVE LEGIBILITY] Target not valid!')
			else:
				print('[PERSPECTIVE LEGIBILITY] Target not valid!')

	def updateTrajectory(self, trajectory, target):

		self._trajectory = trajectory
		self._trajectory_length = trajectory.shape[0]
		self._trajectory_dim = trajectory.shape[1]

		vel_matrix = np.zeros((self._trajectory_length + 1, self._trajectory_length))
		vel_matrix[0, 0] = 1
		vel_matrix[-1, -1] = -1
		for i in range(1, self._trajectory_length):
			vel_matrix[i, i - 1] = -1
			vel_matrix[i, i] = 1

		self._traj_K = np.kron(vel_matrix, np.eye(self._trajectory_dim))
		self._traj_e = np.array(
			list([trajectory[0, :]] + [[0]*self._trajectory_dim] * (self._trajectory_length - 2) + [trajectory[-1, :]]))

		self.updateTarget(target)

	def eraseTrajectory(self):

		self._trajectory = None
		self._trajectory_length = 0
		self._trajectory_dim = 0
		self._target = None

	def get_user_pose(self):
		return self._user_pose

	def get_robot_pose(self):
		return self._robot_pose

	def get_target(self):
		return self._target

	def get_targets_prob(self):
		return self._targets_prob

	def get_perspective_matrix(self):
		return self._perspective_matrix

	def get_user_id(self):
		return self._user_id

	#########################################################################
	####                                                                #####
	####         PERSPECTIVE MANIPULATION & LEGIBILITY METHODS          #####
	####                                                                #####
	#########################################################################

	def perspectiveTransformation(self, orig_point=Point()):

		if self._ros_active and self._using_ros:

			try:
				# Create TF message for point
				point_world = PointStamped()
				point_world.header.stamp = rospy.Time(0)
				point_world.header.frame_id = '/base_link'
				point_world.point.x = orig_point.x
				point_world.point.y = orig_point.y
				point_world.point.z = orig_point.z

				# Transform point from robot space to user space using TFs
				point_user_tf = self._tf_listener.transformPoint('/' + str(self._user_id), point_world)

				# Apply 2D perspective transformation
				user_point_perspective = np.array([point_user_tf.point.x, point_user_tf.point.y, point_user_tf.point.z])
				return user_point_perspective

			except (ValueError, rospy.ROSSerializationException, tf.LookupException,
					tf.ConnectivityException, tf.ExtrapolationException) as e:
				rospy.logerr('[PERSPECTIVE TRANSFORMATION]: Caught Error: %s' % e)
				return None

		else:

			user_orientation = Quaternion(self._user_pose.orientation[0], self._user_pose.orientation[1],
										  self._user_pose.orientation[2], self._user_pose.orientation[3])
			user_position = Point(self._user_pose.position[0], self._user_pose.position[1], self._user_pose.position[2])

			robot_orientation = Quaternion(self._robot_pose.orientation[0], self._robot_pose.orientation[1],
										  self._robot_pose.orientation[2], self._robot_pose.orientation[3])
			robot_position = Point(self._robot_pose.position[0], self._robot_pose.position[1],
								   self._robot_pose.position[2])

			# Get robot and user orientation transformations under Euler Angles
			if self._rotation_type.find('euler') != -1:

				user_euler = T.euler_from_quaternion((user_orientation.x, user_orientation.y,
													  user_orientation.z, user_orientation.w))
				robot_euler = T.euler_from_quaternion((robot_orientation.x, robot_orientation.y,
													   robot_orientation.z, robot_orientation.w))

				user_transformation = T.euler_matrix(user_euler[0], user_euler[1], user_euler[2])

				robot_transformation = T.euler_matrix(robot_euler[0], robot_euler[1], robot_euler[2])

			# Get robot and user orientation transformations under Quaternions
			elif self._rotation_type.find('quaternion') != -1:

				user_transformation = T.quaternion_matrix((user_orientation.x, user_orientation.y,
														   user_orientation.z, user_orientation.w))
				robot_transformation = T.quaternion_matrix((robot_orientation.x, robot_orientation.y,
															robot_orientation.z, robot_orientation.w))

			else:
				print('Invalid rotation type, impossible to transform points')
				return None

			# Add translation of user and robot to transformation matrix
			robot_transformation[0, 3] = robot_position.x
			robot_transformation[1, 3] = robot_position.y
			robot_transformation[2, 3] = robot_position.z
			robot_transformation = np.linalg.inv(robot_transformation)

			user_transformation[0, 3] = user_position.x
			user_transformation[1, 3] = user_position.y
			user_transformation[2, 3] = user_position.z
			user_transformation = np.linalg.inv(user_transformation)

			# Transform point from robot space to user space
			point_world = np.array([[orig_point.x], [orig_point.y], [orig_point.z], [1]])

			# Apply 2D perspective transformation
			user_point_perspective = user_transformation.dot(point_world)

			return user_point_perspective

	def viewportTransformation(self, orig_point=Point()):

		user_point_perspective = self.perspectiveTransformation(orig_point)
		user_point_viewport = self._perspective_matrix.T.dot(user_point_perspective)
		if user_point_viewport[-1] != 1:
			user_point_viewport_return = user_point_viewport / user_point_viewport[-1]
		else:
			user_point_viewport_return = user_point_viewport

		user_point_viewport_return *= 10

		return user_point_viewport_return.reshape((1, len(user_point_viewport))), user_point_viewport

	def getTransformationMatrices(self):

		if self._ros_active and self._using_ros:

			try:
				(trans, rot) = self._tf_listener.lookupTransform('/' + str(self._user_id), '/map', rospy.Time(0))

			except (ValueError, rospy.ROSSerializationException, tf.LookupException, tf.ConnectivityException,
					tf.ExtrapolationException) as e:
				rospy.logerr('[PERSPECTIVE TRANSFORMATION]: Caught Error: %s' % e)
				return None

			user_transformation = T.quaternion_matrix((rot[0], rot[1], rot[2], rot[3]))
			user_transformation[0, 3] = trans[0]
			user_transformation[1, 3] = trans[1]
			user_transformation[2, 3] = trans[2]

		else:

			user_orientation = Quaternion(self._user_pose.orientation[0], self._user_pose.orientation[1],
										  self._user_pose.orientation[2], self._user_pose.orientation[3])
			user_position = Point(self._user_pose.position[0], self._user_pose.position[1], self._user_pose.position[2])

			if self._rotation_type.find('euler') != -1:

				user_euler = T.euler_from_quaternion((user_orientation.x, user_orientation.y,
													  user_orientation.z, user_orientation.w))

				user_transformation = T.euler_matrix(user_euler[0], user_euler[1], user_euler[2])

			elif self._rotation_type.find('quaternion') != -1:

				user_transformation = T.quaternion_matrix((user_orientation.x, user_orientation.y,
														   user_orientation.z, user_orientation.w))

			else:
				print('Invalid rotation type, impossible to transform points')
				return None

			user_transformation[0, 3] = user_position.x
			user_transformation[1, 3] = user_position.y
			user_transformation[2, 3] = user_position.z
			# print(user_transformation)
			user_transformation = np.linalg.inv(user_transformation)

		return user_transformation, self._perspective_matrix

	def getCostMatrices(self, orig_trajectory=None):

		if orig_trajectory is None:
			if self._transformed_trajectory is not None:
				trajectory = self._transformed_trajectory
			else:
				if self._using_ros and self._ros_active:
					rospy.logerr('[COST MATRICES] No trajectory defined!')
				else:
					print('[COST MATRICES] No trajectory defined!')
				return None
		else:
			trajectory = orig_trajectory

		traj_len, traj_dim = trajectory.shape
		vel_matrix = np.zeros((traj_len + 1, traj_len))
		vel_matrix[0, 0] = 1
		vel_matrix[-1, -1] = -1
		for i in range(1, traj_len):
			vel_matrix[i, i - 1] = -1
			vel_matrix[i, i] = 1
		if traj_len > 1:
			e = np.zeros((traj_len + 1, traj_dim))
			e[0] = -trajectory[0, 0:traj_dim]
			e[-1] = trajectory[-1, 0:traj_dim]
		else:
			e = -trajectory[:traj_len+1]

		return vel_matrix, e

	def transformTrajectory(self, trajectory=None, trajectory_dim=None, viewport=True):

		# Choose trajectory to use
		if trajectory is None:
			if not self._trajectory or self._trajectory_length < 0:
				if self._using_ros and self._ros_active:
					rospy.logerr('[TRAJECTORY TO PERSPECTIVE] No trajectory defined!')
				else:
					print('[TRAJECTORY TO PERSPECTIVE] No trajectory defined!')
				return None

			else:
				trajectory = self._trajectory

		if trajectory_dim is None:
			_, trajectory_dim = trajectory.shape

		transformed_trajectory = []

		if viewport:
			user_perspective_trajectory = []
			# compute viewport transformation
			for i in range(len(trajectory)):
				if trajectory_dim < 3:
					user_viewport_ptr, user_ptr = self.viewportTransformation(Point(trajectory[i, 0], trajectory[i, 1], 0))
					transformed_trajectory += [user_viewport_ptr[0, :trajectory_dim]]
					user_perspective_trajectory += [user_ptr[0, :trajectory_dim].reshape(trajectory_dim)]
				else:
					user_viewport_ptr, user_ptr = self.viewportTransformation(Point(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]))
					transformed_trajectory += [user_viewport_ptr[0]]
					user_perspective_trajectory += [user_ptr.reshape(len(user_ptr))]

		else:
			# compute perspective transformation
			for i in range(len(trajectory)):
				if trajectory_dim < 3:
					user_ptr = self.perspectiveTransformation(Point(trajectory[i, 0], trajectory[i, 1], 0))
					transformed_trajectory += [user_ptr[0, :trajectory_dim].reshape(trajectory_dim)[:-1]]
				else:
					user_ptr = self.perspectiveTransformation(Point(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]))
					transformed_trajectory += [user_ptr.reshape(len(user_ptr))[:-1]]

		if viewport:
			return np.array(transformed_trajectory), np.array(user_perspective_trajectory)
		else:
			return np.array(transformed_trajectory)

	def velocityCost(self, trajectory):

		input_len, input_dim = trajectory.shape
		if input_len > 2:
			vel_matrix = np.zeros((input_len - 1, input_len - 2))
			vel_matrix[0, 0] = 1
			vel_matrix[-1, -1] = -1
			for i in range(1, input_len - 2):
				vel_matrix[i, i - 1] = -1
				vel_matrix[i, i] = 1
			e = np.array(list([-trajectory[0, :]]) + [[0]*input_dim]*(input_len - 3) + list([trajectory[-1, :]]))

			k = np.kron(vel_matrix, np.eye(input_dim))
			A = np.dot(k.T, k)
			b = np.dot(k.T, e.ravel())
			c = np.dot(e.ravel().T, e.ravel()) / 2

			A_term = np.dot(np.dot(trajectory[1:-1].ravel().T, A), trajectory[1:-1].ravel())
			b_term = np.dot(trajectory[1:-1].ravel().T, b)
			cost = float(1 / 2.0) * A_term + b_term + c # * 10

		else:
			if input_len > 1:
				vel_matrix = -np.ones(input_len - 1)
				e = np.array(list([trajectory[-1, :]]))

				k = np.kron(vel_matrix, np.eye(input_dim))
				A = np.dot(k.T, k)
				b = np.dot(k.T, e.ravel())
				c = np.dot(e.ravel().T, e.ravel()) / 2

				A_term = np.dot(np.dot(trajectory[0].ravel().T, A), trajectory[0].ravel())
				b_term = np.dot(trajectory[0].ravel().T, b)

			else:
				vel_matrix = np.zeros(1)
				e = -trajectory[:input_len + 1] * 0

				k = np.kron(vel_matrix, np.eye(input_dim))
				A = np.dot(k.T, k)
				b = np.dot(k.T, e.ravel())
				c = np.dot(e.ravel().T, e.ravel()) / 2

				A_term = np.dot(np.dot(trajectory.ravel().T, A), trajectory.ravel())
				b_term = np.dot(trajectory.ravel().T, b)

			cost = float(1 / 2.0) * A_term + b_term + c # * 10

		return cost

	def velocityGrad(self, trajectory):

		input_len, input_dim = trajectory.shape
		if input_len > 2:
			vel_matrix = np.zeros((input_len - 1, input_len - 2))
			vel_matrix[0, 0] = 1
			vel_matrix[-1, -1] = -1
			for i in range(1, input_len - 2):
				vel_matrix[i, i - 1] = -1
				vel_matrix[i, i] = 1
			e = np.array(list([-trajectory[0, :]]) + [[0]*input_dim] * (input_len - 3) + list([trajectory[-1, :]]))

			k = vel_matrix
			A = np.dot(k.T, k)
			b = np.dot(k.T, e)

			A_term = np.dot(A.T, trajectory[1:-1])
			b_term = b

			traj_grad = np.sum(A_term + b_term, axis=0) # * 1000

		else:
			if input_len > 1:
				vel_matrix = -np.ones(input_len - 1)
				e = np.array(list([trajectory[-1, :]]))

				k = vel_matrix
				A = np.dot(k.T, k)
				b = np.dot(k.T, e)

				A_term = np.dot(A.T, trajectory[0])
				b_term = b

			else:
				vel_matrix = np.zeros(1)
				e = -trajectory[:input_len + 1] * 0

				k = vel_matrix
				A = np.dot(k.T, k)
				b = np.dot(k.T, e)

				A_term = np.dot(A.T, trajectory.ravel())
				b_term = b

			traj_grad = A_term + b_term # * 1000

		return traj_grad

	def trajectoryCosts(self, orig_trajectory=None, has_transform=False):

		if orig_trajectory is None:
			if not self._trajectory:
				if self._using_ros and self._ros_active:
					rospy.logerr('[PERSPECTIVE LEGIBILITY] No trajectory defined!')
				else:
					print('[PERSPECTIVE LEGIBILITY] No trajectory defined!')

				return None

			else:
				trajectory = self._trajectory.copy()

		else:
			trajectory = orig_trajectory.copy()

		trajectory_length, trajectory_dim = trajectory.shape
		trajectory_costs = np.zeros(trajectory_length)
		best_trajectory_costs = np.zeros(trajectory_length)
		if has_transform:
			h_trajectory = np.hstack((trajectory, np.ones((trajectory_length, 1))))

		for i in range(trajectory_length):

			# Only in the second point there is a cost for executing the trajectory
			if i > 0:

				# If the trajectory has to be transformed to the user's perspective
				if has_transform:
					# Get perspective and reference transformation matrices and current homogeneous trajectory
					transform_matrix, perspective_matrix = self.getTransformationMatrices()
					temp_traj = h_trajectory[0:i + 1].copy()

					# Project trajectory to user perspective
					transform_traj = temp_traj.dot(transform_matrix.T)
					transform_traj = transform_traj.dot(perspective_matrix)

					# Trajectory normalization
					transform_traj = transform_traj / transform_traj[:, -1, None]
					if np.all(transform_traj[:, -2] < 0):
						transform_traj = transform_traj[:, :-2] / -transform_traj[:, -2, None]
					else:
						transform_traj = transform_traj[:, :-2]
					transform_traj *= 10

					# Compute transfomed trajectory cost
					cost_prev = self.velocityCost(transform_traj)

				else:
					# Get current trajectory executed
					traj_prev = trajectory[0:i + 1]

					# Compute trajectory cost
					cost_prev = self.velocityCost(traj_prev)

			# in the first point there is no trajectory executed so there is no cost
			else:
				cost_prev = 0

			trajectory_costs[i] = cost_prev

			# compute best trajectory cost
			best_traj = np.linspace(trajectory[i, 0], trajectory[-1, 0],
									num=(trajectory_length - i))[:, None]
			for j in range(1, trajectory_dim):
				best_traj = np.hstack((best_traj, np.linspace(trajectory[i, j], trajectory[-1, j],
															  num=(trajectory_length - i))[:, None]))

			if i < trajectory_length - 1:
				if has_transform:
					best_traj = np.hstack((best_traj, np.ones((trajectory_length - i, 1))))
					transform_matrix, perspective_matrix = self.getTransformationMatrices()

					transform_traj = best_traj.dot(transform_matrix.T)
					transform_traj = transform_traj.dot(perspective_matrix)

					# Trajectory normalization
					transform_traj = transform_traj / transform_traj[:, -1, None]
					if np.all(transform_traj[:, -2] < 0):
						transform_traj = transform_traj[:, :-2] / -transform_traj[:, -2, None]
					else:
						transform_traj = transform_traj[:, :-2]
					transform_traj *= 10

					# Compute best trajectory cost
					cost_best = self.velocityCost(transform_traj)

				else:
					# compute best cost
					cost_best = self.velocityCost(best_traj)

			else:
				cost_best = 0

			best_trajectory_costs[i] = cost_best

		return trajectory_costs, best_trajectory_costs

	def trajectory2DProjectGrad(self, orig_trajectory=None):

		if orig_trajectory is None:
			if not self._trajectory:
				if self._using_ros and self._ros_active:
					rospy.logerr('[PERSPECTIVE LEGIBILITY] No trajectory defined!')
				else:
					print('[PERSPECTIVE LEGIBILITY] No trajectory defined!')

				return None

			else:
				trajectory = self._trajectory

		else:
			trajectory = orig_trajectory

		trajectory_len, trajectory_dim = trajectory.shape
		w_scale_factor = 1 / math.tan((self._w_fov/2.0) * (math.pi/180.0))
		h_scale_factor = 1 / math.tan((self._h_fov / 2.0) * (math.pi / 180.0))
		far_plane = self._clip_planes[1]
		near_plane = self._clip_planes[0]
		near_factor = - far_plane / float(far_plane - near_plane)
		far_factor = - (far_plane * near_plane) / float(far_plane - near_plane)

		projection_grad = np.zeros((trajectory_len, 2, 3))
		perspective_traj = self.transformTrajectory(trajectory, viewport=False)

		for i in range(trajectory_len):

			traj_point = perspective_traj[i, :]
			point_grad = np.zeros((2, 3))
			point_grad[0, 0] += w_scale_factor/(near_factor * far_factor * traj_point[2]**2 - far_factor * traj_point[2])
			point_grad[0, 2] += ((w_scale_factor * traj_point[0] * (2 * near_factor * traj_point[2] - 1)) /
							   (near_factor**2 * far_factor * traj_point[2]**4 -
								2 * near_factor * far_factor * traj_point[2]**3 +
								far_factor * traj_point[2]**2))
			point_grad[1, 1] += h_scale_factor/(near_factor * far_factor * traj_point[2]**2 - far_factor * traj_point[2])
			point_grad[1, 2] += ((h_scale_factor * traj_point[1] * (2 * near_factor * traj_point[2] - 1)) /
							   (near_factor**2 * far_factor * traj_point[2]**4 -
								2 * near_factor * far_factor * traj_point[2]**3 +
								far_factor * traj_point[2]**2))

			projection_grad[i] += point_grad

		return projection_grad * 1000

	def trajectoryGradCost(self, orig_trajectory=None):

		if orig_trajectory is None:
			if not self._trajectory:
				if self._using_ros and self._ros_active:
					rospy.logerr('[PERSPECTIVE LEGIBILITY] No trajectory defined!')
				else:
					print('[PERSPECTIVE LEGIBILITY] No trajectory defined!')

				return None

			else:
				trajectory = self._trajectory

		else:
			trajectory = orig_trajectory

		trajectory_length, trajectory_dim = trajectory.shape
		transform_matrix, perspective_matrix = self.getTransformationMatrices()
		grad_costs = np.zeros((trajectory_length, trajectory_dim-1))
		for i in range(trajectory_length):

			best_traj = np.linspace(orig_trajectory[i, 0], orig_trajectory[-1, 0],
									num=(trajectory_length - i))[:, None]
			for j in range(1, trajectory_dim):
				best_traj = np.hstack((best_traj, np.linspace(orig_trajectory[i, j], orig_trajectory[-1, j],
															  num=(trajectory_length - i))[:, None]))

			best_traj = np.hstack((best_traj, np.ones((trajectory_length - i, 1))))
			best_traj = best_traj.dot(transform_matrix.T).dot(perspective_matrix)
			best_traj = best_traj / best_traj[:, -1, None]
			if np.all(best_traj[:, -2] < 0):
				best_traj = best_traj[:, :-2] / -best_traj[:, -2, None]
			else:
				best_traj = best_traj[:, :-2]
			best_traj *= 10

			if i < trajectory_length - 1:
				grad_cost = self.velocityGrad(best_traj)
			else:
				grad_cost = np.zeros(trajectory_dim-1)
			grad_costs[i] = grad_cost

		return grad_costs

	def trajectoryLegibility(self, targets, orig_trajectory=None, costs=None, has_transform=False):

		if self._target is None:
			if self._using_ros and self._ros_active:
				rospy.logerr('[TRAJECTORY LEGIBILITY] No target defined!')
			else:
				print('[TRAJECTORY LEGIBILITY] No target defined!')

			return None

		if orig_trajectory is None:
			if not self._trajectory or self._trajectory_length < 0:
				if self._using_ros and self._ros_active:
					rospy.logerr('[TRAJECTORY LEGIBILITY] No trajectory defined!')
				else:
					print('[TRAJECTORY LEGIBILITY] No trajectory defined!')

				return None

			else:
				# transform class defined trajectory to perspective view and compute costs
				trajectory = self._trajectory.copy()
				trajectory_costs, best_trajectory_costs = self.trajectoryCosts(trajectory, has_transform=has_transform)

		else:
			if costs is None or len(costs) < 3:
				# transform given trajectory to perspective view and compute costs
				trajectory = orig_trajectory.copy()
				trajectory_costs, best_trajectory_costs = self.trajectoryCosts(trajectory, has_transform=has_transform)

			else:
				trajectory = orig_trajectory.copy()
				trajectory_costs = costs[0, :]
				best_trajectory_costs = costs[2, :]

		trajectory_length, trajectory_dim = trajectory.shape
		target_idx = self._targets_prob.keys().index(self._target)
		targets_keys = self._targets_prob.keys()
		costs_targets = {}

		for i in range(len(targets_keys)):

			if i != target_idx:

				trajectory_tmp = trajectory.copy()
				trajectory_tmp[-1] = targets[i]

				target_costs, best_target_costs = self.trajectoryCosts(trajectory_tmp, has_transform=has_transform)

				costs_targets[targets_keys[i]] = [target_costs, best_target_costs]

			else:
				costs_targets[targets_keys[i]] = [trajectory_costs, best_trajectory_costs]

		prob_target_traj_targets = {}
		for i in range(len(targets_keys)):

			target_costs = costs_targets[targets_keys[i]][0]
			best_target_costs = costs_targets[targets_keys[i]][1]

			prob_target_traj = (self._targets_prob[targets_keys[i]] *
								(np.exp(-target_costs[:] - best_target_costs[:])) /
								np.exp(-best_target_costs[0]))

			prob_target_traj_targets[targets_keys[i]] = prob_target_traj

		partial_trajectory_legibility = (prob_target_traj_targets[self._target] *
										 1 / np.array(prob_target_traj_targets.values()).sum(axis=0))
		time_function = np.array([(trajectory_length - i) / float(trajectory_length) for i in range(trajectory_length)])

		# full trajectory legibility
		traj_legibility = np.sum(partial_trajectory_legibility*time_function) / np.sum(time_function)

		return traj_legibility

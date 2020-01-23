# !/usr/bin/env python
import numpy as np
import rosgraph
import tf
import rospy
import tf.transformations as T

from geometry_msgs.msg import Point, PointStamped, Quaternion, Pose

np.set_printoptions(precision=3, linewidth=200, threshold=2000)


class UserPerspectiveLegibility(object):

	def __init__(self, using_ros, user_pose, robot_pose, orientation_type, user_id,
	             target=None, targets_prob=None, perspective_matrix=np.eye(3)):

		if targets_prob is None:
			self._targets_prob = {'A': 0.5, 'B': 0.5}
		else:
			self._targets_prob = targets_prob

		self._target = target
		self._user_angle = 0.0
		self._user_pose = user_pose
		self._rotation_type = orientation_type
		self._using_ros = using_ros
		self._user_id = user_id
		self._perspective_matrix = perspective_matrix
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
			list([trajectory[0, :]] + [[0, 0, 0]] * (self._trajectory_length - 2) + [trajectory[-1, :]]))

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
				#point_world = self._tf_listener.transformPoint('/map', point_robot)
				point_user_tf = self._tf_listener.transformPoint('/' + str(self._user_id), point_world)

				# Apply 2D perspective transformation
				point_user = np.array([point_user_tf.point.x, point_user_tf.point.y, point_user_tf.point.z])
				user_point_perspective = self._perspective_matrix.dot(point_user)
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

			user_transformation[0, 3] = user_position.x
			user_transformation[1, 3] = user_position.y
			user_transformation[2, 3] = user_position.z

			# Transform point from robot space to user space
			#robot_transformation = np.linalg.inv(robot_transformation)
			point_world = np.array([[orig_point.x], [orig_point.y], [orig_point.z], [1]])
			#point_world = robot_transformation.dot(point_robot)

			# Apply 2D perspective transformation
			point_user = user_transformation.dot(point_world)
			user_point_perspective = self._perspective_matrix.dot(point_user[:-1])
			#print(user_point_perspective)
			return user_point_perspective.reshape((1, len(user_point_perspective)))

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

		return user_transformation, self._perspective_matrix

	def getCostMatrices(self, i_traj_init, i_traj_end, orig_trajectory=None):

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

		traj_len = i_traj_end - i_traj_init + 1
		vel_matrix = np.zeros((traj_len + 1, traj_len))
		vel_matrix[0, 0] = 1
		vel_matrix[-1, -1] = -1
		for i in range(1, traj_len):
			vel_matrix[i, i - 1] = -1
			vel_matrix[i, i] = 1
		if traj_len > 1:
			e = np.array(list(-trajectory[i_traj_init]) + [0, 0, 0] * (traj_len - 1) + list(trajectory[i_traj_end]))
		else:
			e = -trajectory[i_traj_init]

		return vel_matrix, e

	def transformTrajectory(self, trajectory=None):

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

		transformed_trajectory = []

		# compute perspective transformation
		for i in range(len(trajectory)):
			user_ptr = self.perspectiveTransformation(Point(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]))
			transformed_trajectory += [user_ptr[0]]

		return np.array(transformed_trajectory)

	def trajectoryCosts(self, orig_trajectory=None, has_transform=False):

		if orig_trajectory is None:
			if not self._trajectory:
				if self._using_ros and self._ros_active:
					rospy.logerr('[PERSPECTIVE LEGIBILITY] No trajectory defined!')
				else:
					print('[PERSPECTIVE LEGIBILITY] No trajectory defined!')

				return None

			else:
				self._transformed_trajectory = self.transformTrajectory(self._trajectory)
				trajectory = self._transformed_trajectory

		else:
			if has_transform:
				self._trajectory = orig_trajectory.copy()
				self._transformed_trajectory = self.transformTrajectory(orig_trajectory)
				trajectory = self._transformed_trajectory

			else:
				trajectory = orig_trajectory

		trajectory_length, trajectory_dim = trajectory.shape
		trajectory_costs = np.zeros((trajectory_length, 2))
		best_trajectory_costs = np.zeros(trajectory_length)
		for i in range(trajectory_length):

			#print('\n\nPoint: ' + str(i))
			# get aux cost matrices for trajectory before and after current point
			if i == (trajectory_length - 1):
				vel_matrix_prev, e_prev = self.getCostMatrices(0, i, trajectory)
				vel_matrix_after = np.zeros(1)
				e_after = np.array(trajectory[0])*0

			# in the last point the trajectoy cost is the same before and after
			elif i > 0:
				vel_matrix_prev, e_prev = self.getCostMatrices(0, i, trajectory)
				vel_matrix_after, e_after = self.getCostMatrices(i, trajectory_length-1, trajectory)

			# in the first point there is no trajectory cost for the trajectory before it
			else:
				vel_matrix_prev = np.zeros(1)
				e_prev = np.array(trajectory[0])*0

				vel_matrix_after, e_after = self.getCostMatrices(0, trajectory_length-1, trajectory)

			# compute partial trajectory cost
			k_prev = np.kron(vel_matrix_prev, np.eye(trajectory_dim))
			traj_prev = trajectory[0:i+1].ravel()
			A_prev = k_prev.T.dot(k_prev)
			b_prev = k_prev.T.dot(e_prev)
			c_prev = e_prev.T.dot(e_prev) / 2
			cost_prev = float(1 / 2.0) * traj_prev.dot(A_prev).dot(traj_prev.T) + traj_prev.dot(b_prev) + c_prev

			k_after = np.kron(vel_matrix_after, np.eye(trajectory_dim))
			traj_after = trajectory[i:].ravel()
			A_after = k_after.T.dot(k_after)
			b_after = k_after.T.dot(e_after)
			c_after = e_after.T.dot(e_after) / 2
			cost_after = float(1 / 2.0) * traj_after.dot(A_after).dot(traj_after.T) + traj_after.dot(b_after) + c_after

			trajectory_costs[i, 0] = cost_prev
			trajectory_costs[i, 1] = cost_after

			# compute best trajectory cost
			best_traj = np.linspace(trajectory[i, 0], trajectory[-1, 0],
			                        num=(trajectory_length - i))[:, None]
			for j in range(1, trajectory_dim):
				best_traj = np.hstack((best_traj, np.linspace(trajectory[i, j], trajectory[-1, j],
				                                              num=(trajectory_length - i))[:, None]))

			best_traj = best_traj.ravel()

			if i < trajectory_length - 1:
				cost_best = float(1 / 2.0) * best_traj.dot(A_after).dot(best_traj.T) + best_traj.dot(b_after) + c_after
			else:
				cost_best = 0
			best_trajectory_costs[i] = cost_best

		#print(trajectory_costs)
		#print(best_trajectory_costs)

		return trajectory_costs, best_trajectory_costs

	def trajectoryGradCost(self, orig_trajectory=None, has_transform=False):

		if orig_trajectory is None:
			if not self._transformed_trajectory:
				if self._using_ros and self._ros_active:
					rospy.logerr('[PERSPECTIVE LEGIBILITY] No trajectory defined!')
				else:
					print('[PERSPECTIVE LEGIBILITY] No trajectory defined!')

				return None

			else:
				self._transformed_trajectory = self.transformTrajectory(self._trajectory)
				trajectory = self._transformed_trajectory

		else:
			if has_transform:
				self._trajectory = orig_trajectory.copy()
				self._transformed_trajectory = self.transformTrajectory(orig_trajectory)
				trajectory = self._transformed_trajectory

			else:
				trajectory = orig_trajectory

		trajectory_length, trajectory_dim = trajectory.shape
		best_trajectory_costs = np.zeros(trajectory_length)
		for i in range(len(trajectory)):
			#print('\n')

			if i == trajectory_length - 1:
				vel_matrix = np.zeros(1)
				e = np.array(trajectory[0])*0

			elif i > 0:
				vel_matrix, e = self.getCostMatrices(i, trajectory_length - 1, trajectory)

			else:
				vel_matrix, e = self.getCostMatrices(0, trajectory_length - 1, trajectory)

			k_matrix = np.kron(vel_matrix, np.eye(trajectory_dim))
			A = k_matrix.T.dot(k_matrix)
			print(A)
			b = k_matrix.T.dot(e)

			best_traj = np.linspace(trajectory[i, 0], trajectory[-1, 0],
			                        num=(trajectory_length - i))[:, None]
			for j in range(1, trajectory_dim):
				best_traj = np.hstack((best_traj, np.linspace(trajectory[i, j], trajectory[-1, j],
				                                              num=(trajectory_length - i))[:, None]))

			best_traj = best_traj.ravel()
			if i < trajectory_length - 1:
				cost_best = np.sum(best_traj.dot(A) + b)
				#print(A)
				#print(b)
				#print(best_traj.dot(A))
				#print(best_traj.dot(A) + b)
				#print(np.sum(best_traj.dot(A) + b))
			else:
				cost_best = 0
			best_trajectory_costs[i] = cost_best

		#print('gradients')
		#print(best_trajectory_costs)
		#print('\n')

		return best_trajectory_costs

	def trajectoryLegibility(self, orig_trajectory=None, costs=None, has_transform=False):

		if self._target is None:

			if self._using_ros and self._ros_active:
				rospy.logerr('[PERSPECTIVE LEGIBILITY] No target defined!')
			else:
				print('[PERSPECTIVE LEGIBILITY] No target defined!')

			return None

		if orig_trajectory is None:
			if not self._trajectory or self._trajectory_length < 0:
				if self._using_ros and self._ros_active:
					rospy.logerr('[PERSPECTIVE LEGIBILITY] No trajectory defined!')
				else:
					print('[PERSPECTIVE LEGIBILITY] No trajectory defined!')

				return None

			else:
				# transform class defined trajectory to perspective view and compute costs
				self._transformed_trajectory = self.transformTrajectory()
				trajectory_costs, best_trajectory_costs = self.trajectoryCosts()

		else:
			if costs is None or len(costs) < 3:
				# transform given trajectory to perspective view and compute costs
				if has_transform:
					self._transformed_trajectory = self.transformTrajectory(orig_trajectory)
					trajectory = self._transformed_trajectory
				else:
					trajectory = orig_trajectory
				trajectory_costs, best_trajectory_costs = self.trajectoryCosts(trajectory)

			else:
				trajectory_costs = costs[0:1, :]
				best_trajectory_costs = costs[2, :]

		# compute partial legibility values
		prob_target_traj = (self._targets_prob[self._target] *
		                    (np.exp(-trajectory_costs[:, 0] - best_trajectory_costs[:])) /
		                    np.exp(-best_trajectory_costs[0])) * 1 / float(len(self._targets_prob.keys()))
		time_function = np.array([(self._trajectory_length - i) / float(self._trajectory_length)
		                          for i in range(self._trajectory_length)])
		partial_trajectory_legibility = prob_target_traj * time_function

		# full trajectory legibility
		traj_legibility = np.sum(partial_trajectory_legibility) / np.sum(time_function)

		return traj_legibility

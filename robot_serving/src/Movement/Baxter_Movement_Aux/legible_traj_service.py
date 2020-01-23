import numpy as np
import rospy
import scipy.io as sio
from promp.copromp import CoProMP
from promp.utils.utils import linear_phase
from promp.utils.utils import normalized_gaussian_basis

from robot_serving.srv import *
from Baxter_Movement_Aux.baxter_connection import BaxterConnection, GoalStatus, baxter_interface
from Baxter_Movement_Aux.trajectory_executor import TrajectoryExecutor
from std_msgs.msg import Int32


class LegibleTrajectoryServer(object):

	def __init__(self):

		self._service_server = rospy.Service("movement_decision_legible", Movement, self.legible_traj_srv_handler)
		self._Y = []
		self._O = []
		self.load_trajs()

		"""
		rospy.loginfo("Creating COPMP")
		self._copmp = CoProMP(self._O, self._Y, 3, 7, o_dt=1, dt=0.001, Sigma_y=None)
		self._copmp.build(linear_phase, lambda z, dt: normalized_gaussian_basis(2, z, dt),
						  linear_phase, lambda z, dt: normalized_gaussian_basis(10, z, dt))
		rospy.loginfo("COPMP created")
		"""

		self._baxter_connect = BaxterConnection()

		self._log_file = open('log_trajectory.txt', 'w')
		rospy.on_shutdown(self._log_file.close)
		self._can_start = False

		self._start_sginal = rospy.Subscriber('/start', Int32, self.start_handler)

		while not self._can_start:
			pass

		self._start_sginal.unregister()
		self.send_neutral()

		rospy.loginfo("Legible trajectory service server created")

	def send_neutral(self):
		starting_position = {
			'left': {
				'left_s0': 0.5721748332275391,
				'left_s1': 0.2784175126831055,
				'left_w0': 0.13038836682128907,
				'left_w1': 0.730558349395752,
				'left_w2': -1.0507768385009766,
				'left_e0': -1.047708876928711,
				'left_e1': 2.012199296209717
			},
			'right': {
				'right_s0': -0.4460049135681153,
				'right_s1': 0.03873301484985352,
				'right_w0': -0.18292720874633792,
				'right_w1': 0.6446554253723145,
				'right_w2': -1.5370487477050783,
				'right_e0': 1.5604419546936037,
				'right_e1': 2.1698158219848636
			}
		}

		left = baxter_interface.Limb('left')
		right = baxter_interface.Limb('right')

		left.move_to_joint_positions(starting_position['left'])
		right.move_to_joint_positions(starting_position['right'])

	def start_handler(self, msg):
		rospy.loginfo("Handler")
		if msg.data == 1:
			self._can_start = True

	def load_trajs(self):
		N = 20
		legible_trajs = []
		legible_targets = []
		for i in range(1, N + 1):
			print('clean_trajectories/legible_traj%d.mat' % i)
			legible_trajs.append(
					sio.loadmat('./trajectory_recording/clean_trajectories/legible_traj%d.mat' % i)['traj'][:, 1:7 + 1])
			legible_targets.append(
					sio.loadmat('./trajectory_recording/clean_trajectories/legible_traj%d.mat' % i)['target'])

		self._Y = np.hstack(legible_trajs)
		self._O = np.hstack(legible_targets)

	def legible_traj_srv_handler(self, req):
		result = 1
		target = [req.x_pos, req.y_pos, req.z_pos]
		new_o = np.vstack(target)

		rospy.loginfo("Creating COPMP")
		copmp = CoProMP(self._O, self._Y, 3, 7, o_dt=1, dt=0.0001, Sigma_y=None)
		copmp.build(linear_phase, lambda z, dt: normalized_gaussian_basis(2, z, dt),
					linear_phase, lambda z, dt: normalized_gaussian_basis(10, z, dt))
		rospy.loginfo("COPMP created")

		rospy.loginfo("Conditioning COPMP to target")
		copmp.condition(new_o, 1)
		ymp = copmp.most_probable()

		time = ymp[:, 0][:, np.newaxis]
		right_traj = ymp[:, 1:7 + 1]

		rospy.loginfo("Executing legible trajectory")
		te = TrajectoryExecutor(time, 10, None, right_traj)
		te.execute()

		rospy.loginfo("Legible trajectory executed")

		return MovementResponse(result)

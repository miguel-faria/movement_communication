import math

import baxter_interface
import numpy as np
import rospy
import scipy.io as sio
from promp.copromp import CoProMP
from promp.utils.utils import linear_phase
from promp.utils.utils import normalized_gaussian_basis

# import do teu state extractor
from robot_serving.msg import Cups
from Baxter_Movement_Aux.trajectory_executor import TrajectoryExecutor

_target = []

def neutral():
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

def get_pos_callback(msg):
	if msg.cups_pos_x > 0:
		global _target

		dist = math.sqrt(math.pow(msg.cups_pos_x[0], 2) + np.math.pow(msg.cups_pos_y[0], 2) + math.pow(msg.cups_pos_z[0], 2))
		rospy.loginfo("Target Pos: (" + str(msg.cups_pos_x[0]) + ", " + str(msg.cups_pos_y[0]) + ", " + str(msg.cups_pos_z[0]) + ")")
		rospy.loginfo("Dist: " + str(dist))
		rospy.loginfo("\n\n\n")
		if dist < float("inf"):
			_target = [msg.cups_pos_x[0], msg.cups_pos_y[0], msg.cups_pos_z[0]]

def init_ros_vars():
    # Init ROS and baxter stuff
    rospy.init_node('cup_filling_task_recording')
    rospy.loginfo('cup_filling_task_recording: init: Node initialized.')
    rospy.Subscriber('/vision_processing/cups_pub', Cups, get_pos_callback)


def init_baxter_vars():
    limbs = {'left': baxter_interface.Limb('left'),
             'right': baxter_interface.Limb('right')}
    return limbs


def load_trajs():
	N = 20
	predict_trajs = []
	predict_targets = []
	legible_trajs = []
	legible_targets = []
	for i in range(1, N + 1):
		print('clean_trajectories/predict_traj%d.mat' % i)
		predict_trajs.append(sio.loadmat('../../trajectory_recording/clean_trajectories/predict_traj%d.mat' % i)['traj'][:, 1:7+1])
		predict_targets.append(sio.loadmat('../../trajectory_recording/clean_trajectories/predict_traj%d.mat' % i)['target'])
		print('clean_trajectories/legible_traj%d.mat' % i)
		legible_trajs.append(sio.loadmat('../../trajectory_recording/clean_trajectories/legible_traj%d.mat' % i)['traj'][:, 1:7+1])
		legible_targets.append(sio.loadmat('../../trajectory_recording/clean_trajectories/legible_traj%d.mat' % i)['target'])

	Ypredict = np.hstack(predict_trajs)
	Ylegible = np.hstack(legible_trajs)
	Opredict = np.hstack(predict_targets)
	Olegible = np.hstack(legible_targets)

	return Ypredict, Opredict, Ylegible, Olegible


def main():
	# Load cleaned trajectories and respective targets
	# Y = [y1_x y1_y y1_z, ..., y2_x, y2_y, y2_z, ...]
	# O = [o1_x, o1_y, o1_z, ...]
	Y, O, _, _ = load_trajs()

	init_ros_vars()
	neutral()

	rospy.loginfo("Creating COPMP")
	copmp = CoProMP(O, Y, 3, 7, o_dt=1, dt=0.001, Sigma_y=None)
	copmp.build(linear_phase, lambda z, dt: normalized_gaussian_basis(2, z, dt),
				linear_phase, lambda z, dt: normalized_gaussian_basis(10, z, dt))
	rospy.loginfo("COPMP created")

	# Use state extractor to obtain new target
	# new_o = ...
	raw_input()
	new_o = np.vstack(_target)

	rospy.loginfo("Getting COPMP trajectory")
	copmp.condition(new_o, 1)
	Ymp = copmp.most_probable()

	# Use trajectory executor to execute the trajectory
	# You shall turn on jtas before running this
	time = Ymp[:, 0][:, np.newaxis]
	right_traj = Ymp[:, 1:7 + 1]

	rospy.loginfo("Executing COPMP")
	te = TrajectoryExecutor(time, 10, None, right_traj)
	te.execute()


if __name__ == '__main__':
	main()

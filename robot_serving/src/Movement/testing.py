import argparse

import baxter_interface
import numpy as np
import rospy
import scipy.io as sio

from Baxter_Movement_Aux.trajectory_executor import TrajectoryExecutor


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


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        dest='input', 
                        default='traj',
                        help='Name of input file. (default: traj)')
    args = parser.parse_args()

    rospy.init_node('test_exec')

    neutral()
    
    mat = sio.loadmat(args.input)

    time = mat['traj'][:, 0][:, np.newaxis]
    right_traj = mat['traj'][:, 1:7+1]

    te = TrajectoryExecutor(time, 10, None, right_traj)
    te.execute()

if __name__ == '__main__':
    main()

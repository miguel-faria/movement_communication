import rospy
import numpy as np
import scipy.io as sio
import baxter_interface
import argparse
from symab_execution.trajectory_executor import TrajectoryExecutor


def neutral():
    starting_position = {
        'left': {
            'left_s0': -0.08,
            'left_s1': -1.0,
            'left_w0':  0.67,
            'left_w1':  1.03,
            'left_w2': -0.50,
            'left_e0': -1.19,
            'left_e1':  1.94
        },
        'right': {
            'right_s0':  0.08,
            'right_s1': -1.0,
            'right_w0': -0.67,
            'right_w1':  1.03,
            'right_w2':  0.5,
            'right_e0':  1.19,
            'right_e1':  1.94
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
    left_traj = mat['traj'][:, 1:7+1]
    right_traj = mat['traj'][:, 15:21+1]

    te = TrajectoryExecutor(time, 10, left_traj, right_traj)
    te.execute()

if __name__ == '__main__':
    main()

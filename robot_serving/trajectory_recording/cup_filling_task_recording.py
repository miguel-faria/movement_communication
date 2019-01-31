import argparse
import signal

import baxter_interface
import rospy
import scipy.io as sio
import math

from trajectory_recorder import TrajectoryRecorder

from robot_serving.msg import Cups

_x_target = float()
_y_target = float()
_z_target = float()
_stop = False


def signal_handler(signum, frame):
    if signum == signal.SIGINT:
        rospy.loginfo("Interruption from user.")
        exit()
    elif signum == signal.SIGSEGV:
        rospy.loginfo("Internal error detected.")
        raise OSError("Caught a SIGSEGV during execution of cup_filling_task_recording.")
    else:
        rospy.loginfo("Unexpected error caught.")
        raise OSError("Caught a " + str(signal.SIGSEGV) + " during execution of cup_filling_task_recording.")


def timer_callback(event):
    global _stop
    if not _stop:
        dist = math.sqrt(math.pow(_x_target, 2) + math.pow(_y_target, 2) + math.pow(_z_target, 2))
        rospy.loginfo("Target Pos: (" + str(_x_target) + ", " + str(_y_target) + ", " + str(_z_target) + ")")
        rospy.loginfo("Dist: " + str(dist))
        rospy.loginfo("\n\n\n")


def get_pos_callback(msg):
    if msg.cups_pos_x > 0:
        global _x_target
        global _y_target
        global _z_target

        _z_target = msg.cups_pos_z[0]
        _y_target = msg.cups_pos_y[0]
        _x_target = msg.cups_pos_x[0]


def init_ros_vars():
    # Init ROS and baxter stuff
    rospy.init_node('cup_filling_task_recording')
    rospy.loginfo('cup_filling_task_recording: init: Node initialized.')
    rospy.Subscriber('/vision_processing/cups_pub', Cups, get_pos_callback)
    rospy.Timer(rospy.Duration(2), timer_callback)


def init_baxter_vars():
    limbs = {'left': baxter_interface.Limb('left'),
             'right': baxter_interface.Limb('right')}
    return limbs


def init_recording_vars():

    rec = TrajectoryRecorder('right', 100)
    return rec


def before_record(limbs):
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

    for limb_name, limb in limbs.items():
        limb.move_to_joint_positions(starting_position[limb_name])


def post_record(traj, target, output_filename):
    sio.savemat(output_filename, {'traj': traj, 'target': target})    
    

def task_recording():

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGSEGV, signal_handler)

    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        help='Name of output file',
                        type=str,
                        required=True)
    args = parser.parse_args()
    output_filename = args.output

    init_ros_vars()
    rospy.loginfo('\ncup_filling_task_recording: main: Make sure gripper_control is running.')
    rospy.loginfo('Press Enter when ready.\n')

    raw_input()
    limbs = init_baxter_vars()
    before_record(limbs)

    rospy.loginfo('\ncup_filling_task_recording: main: Place cup in a visible and reachable position.')
    rospy.loginfo('Press Enter to continue.\n')

    raw_input()
    rec = init_recording_vars()
    rospy.loginfo('\ncup_filling_task_recording: main: Wait for target detection to be stable')
    rospy.loginfo('Press Enter to continue.\n')

    raw_input()
    global _stop
    _stop = True
    target = [_x_target, _y_target, _z_target]
    trajectory = rec.record()

    post_record(trajectory, target, output_filename)


if __name__ == '__main__':
    task_recording()

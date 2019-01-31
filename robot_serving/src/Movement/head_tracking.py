import rospy
import argparse

from Baxter_Movement_Aux.Head_Tracker import HeadTracker


def head_tracking():
    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limb',
                        nargs='+',
                        help='Limb to track',
                        type=str,
                        required=True)
    args = parser.parse_args()
    tracking_limb = args.limb

    # Starting Tracking Node
    rospy.init_node('head_tracker')
    ht = HeadTracker(['left', 'right'], tracking_limb)
    rospy.loginfo('Created Head Tracker')
    for limb in tracking_limb:
        ht.add_listener(limb)
    rospy.loginfo('Added Listeners')
    rospy.on_shutdown(ht.head_tracker_shutdown)
    rospy.spin()


if __name__ == "__main__":
    head_tracking()

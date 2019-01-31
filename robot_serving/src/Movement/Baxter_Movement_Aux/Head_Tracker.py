"""
    HEADTRACKER CLASS

    This class is responsible for storing all information for Baxter's Head tracking the arms' movement.
    As with the "Trajectory" class all communications are done over ROS.
"""

import baxter_interface
import rospy

from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import *


class HeadTracker(object):

    TIMER_PERIOD = 3
    NEUTRAL_HEAD_ANGLE = 0.0

    """
        Constructor
    """
    def __init__(self, limbs, tracking_limb):
        # Initializing connection to Baxter Head Control
        rospy.loginfo('Starting Head-Arm Movement Tracker')
        rospy.loginfo('Initializing connection to Baxter head')
        self._done = False
        self._head = baxter_interface.Head()
        rospy.loginfo('Getting Robot State')
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        rospy.loginfo('Enabling Robot')
        self._rs.enable()

        # Initializing arm movement variables
        self._track_limb = tracking_limb
        self._endpoint_topics = dict()
        self._endpoint_pos = dict()
        for limb in limbs:
            self._endpoint_topics[limb] = '/robot/limb/' + limb + '/endpoint_state'
        self._endpoint_listeners = dict()
        self._goal = dict()
        self.head_neutral()
        self.timer = rospy.Timer(rospy.Duration(self.TIMER_PERIOD), self.update_tracking_target(self._track_limb))
        rospy.loginfo('Running Head-Arm Movement Tracker. Ctrl-C to quit')

    """
       Getters and Setters
    """
    @property
    def track_limb(self):
        return self._track_limb

    @track_limb.setter
    def track_limb(self, limb):
        self._track_limb = limb

    @property
    def endpoint_topics(self):
        return self._endpoint_topics

    @endpoint_topics.setter
    def endpoint_topics(self, topics):
        self._endpoint_topics = topics

    @property
    def endpoint_pos(self):
        return self._endpoint_pos

    @endpoint_pos.setter
    def endpoint_pos(self, pos):
        self._endpoint_pos = pos

    @property
    def endpoint_listeners(self):
        return self._endpoint_listeners

    @endpoint_listeners.setter
    def endpoint_listeners(self, listeners):
        self._endpoint_listeners = listeners

    def _get_move_target(self, limb):
        target = dict()
        look_pos = self._endpoint_pos[limb]

        # FIXME TODO - Compute target from grip position

        target['speed'] = 15
        target['timeout'] = 0
        return target

    """
        Modifiers
    """
    def head_tracker_shutdown(self):
        rospy.loginfo('Stopping Head-Arm Movement Tracker')
        self._clean_head_shutdown()
        for key in self._endpoint_listeners:
            self._endpoint_listeners[key].unregister()
        self._endpoint_listeners.clear()
        self._endpoint_topics.clear()
        self._endpoint_pos.clear()
        self._goal = list()

    def _clean_head_shutdown(self):
        if self._done:
            self.head_neutral()
        if not self._init_state and self._rs.state().enabled:
            rospy.loginfo('Disabling Robot...')
            self._rs.disable()

    def _endpoint_position(self, msg, limb):
        self._endpoint_pos[limb] = {'x': msg.pose.position.x,
                                    'y': msg.pose.position.y,
                                    'z': msg.pose.position.z}
        rospy.loginfo('Received new positions')

    def add_listener(self, limb):
        endpoint_topic = self._endpoint_topics[limb]
        if limb in self._endpoint_listeners:
            rospy.loginfo('Endopoint listener for ' + limb + ' limb already exists, replacing old endpoint....')
        self._endpoint_listeners[limb] = rospy.Subscriber(endpoint_topic, EndpointState, self._endpoint_position, limb)
        rospy.loginfo('Endopoint listener for ' + limb + ' limb added at ' + endpoint_topic)

    def remove_listener(self, limb):
        if limb in self._endpoint_listeners:
            self._endpoint_listeners.pop(limb)
            rospy.loginfo('Endpoint listener for ' + limb + ' limb removed.')
        else:
            rospy.loginfo('No listener for the ' + limb + ' limb!!')

    def update_tracking_target(self, limb):
        target = self._get_move_target(limb)
        if (not self._goal) | (self._goal != target):
            self._goal = target
            self.follow_target()
        elif self._goal == target:
            self.stop_head()

    """
        Head Movement public methods
    """
    def head_neutral(self):
        self._head.set_pan(self.NEUTRAL_HEAD_ANGLE)

    def move_head(self, angle=0, speed=50, timeout=0):
        self._head.set_pan(angle, speed, timeout)

    def follow_target(self):
        self._head.set_pan(self._goal['angle'], self._goal['speed'], self._goal['timeout'])

    def stop_head(self):
        self._goal['speed'] = 0
        self._head.set_pan(self._goal['angle'], self._goal['speed'], self._goal['timeout'])

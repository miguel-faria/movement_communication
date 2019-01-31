import numbers
import numpy as np
import baxter_interface
import rospy


class TrajectoryRecorder(object):
    VALID_LIMBS = ['left', 'right', 'both']
    
    def __init__(self, limb='both', rate=100, pre_cb=None, post_cb=None):
        # Arg check
        if not limb in TrajectoryRecorder.VALID_LIMBS:
            raise ValueError('limb %s not valid.' % limb)
        if not isinstance(rate, numbers.Number):
            raise ValueError('rate must be a number.' % limb)
        if pre_cb is not None and not callable(pre_cb):
            raise ValueError('pre_cb must be either None or a function.')
        if post_cb is not None and not callable(post_cb):
            raise ValueError('post_cb must be either None or a function.')
        
        # From limb str to Baxter.Limb
        if limb == 'both':
            self._limbs = {'left': baxter_interface.Limb('left'),
                           'right': baxter_interface.Limb('right')}
        else:
            self._limbs = {limb: baxter_interface.Limb(limb)}

        # Define the cuff sensor callback
        master_limb_name = self._limbs.keys()[0]
        cuff_sensor = baxter_interface.DigitalIO('%s_lower_cuff' % 
                                                 (master_limb_name,))
        cuff_sensor.state_changed.connect(self._cuff_action)

        self._rate = rospy.Rate(rate)
        self._pre_cb = pre_cb
        self._post_cb = post_cb
        self._record = False
        self._done = False

    def record(self):
        if self._pre_cb is not None:
            self._pre_cb()

        print('TrajectoryRecorder: record: Starting to record.')
        traj = self._record_aux()
        print('TrajectoryRecorder: record: Finished recording.')

        if self._post_cb is not None:
            self._post_cb()

        return traj

    def _cuff_action(self, value):
        print('TrajectoryRecorder: _cuff_action: state changed: %s', str(value))

        if value:
            self._record = value
        else:
            self._done = not value

    def _record_aux(self):
        # Wait for record time
        while not self._record:
            pass

        time = rospy.get_time()
        storage = []
        while not self._done:
            tmp_storage = []

            # Grab desired info
            for limb_name, limb in iter(sorted(self._limbs.items())):  # make sure left is first..
                _, angles = zip(*sorted(limb.joint_angles().items()))
                angles = list(angles)  # tuple to list
                # angles = [limb.joint_angle(j) for j in limb.joint_angles()]
                pose = limb.endpoint_pose()
                pose = list(pose['position']) + list(pose['orientation'])
                tmp_storage += angles + list(pose)

            # Current timestamp
            time = rospy.get_time()

            # Store it!
            storage.append([time] + tmp_storage)

            # Sleep
            self._rate.sleep()

        # To numpy and normalize time (start at 0)
        storage = np.vstack(storage)
        storage[:, 0] -= storage[0, 0]

        return storage

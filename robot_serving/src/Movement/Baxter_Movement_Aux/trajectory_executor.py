import numpy as np
import rospy

from Baxter_Movement_Aux.Trajectory import Trajectory

class TrajectoryExecutor(object):
    # time: 2d column array
    def __init__(self,
                 time, duration=1,
                 left_trajectory=None, right_trajectory=None):
        provided_dt = np.average(np.diff(time[:, 0]))
        self._dt = provided_dt * duration

        self._trajectories = {}
        if left_trajectory is not None:
            self._trajectories['left'] = Trajectory('left')
            rospy.on_shutdown(self._trajectories['left'].stop)

            TrajectoryExecutor._add_points(time, self._dt,
                                           self._trajectories['left'],
                                           left_trajectory)
        if right_trajectory is not None:
            self._trajectories['right'] = Trajectory('right')
            rospy.on_shutdown(self._trajectories['right'].stop)

            TrajectoryExecutor._add_points(time, self._dt,
                                           self._trajectories['right'],
                                           right_trajectory)

    def execute(self):

        #Move to target
        traj = self._trajectories['right']
        traj.start()
        traj.wait()
        traj.stop_all()
        Trajectory.send_neutral()


    @staticmethod
    def _add_points(time, dt, trajectory, points):
        for t in range(0, time.shape[0]):
            trajectory.add_point(list(points[t, :]), t * dt)

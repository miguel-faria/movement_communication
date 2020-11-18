# !/usr/bin/env python
try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink
except ImportError:
    import sys
    sys.exit('The "ikpy" Python module is not installed. '
             'Please upgrade "pip" and install ikpy with this command: "pip install ikpy"')


import spatialmath as sm
import tf.transformations as tr
import math
import sys
print(sys.version)
from ropy.robot.ETS import ETS
from pathlib import Path
from geometry_msgs.msg import Pose, Point, Quaternion
from user_perspective_legibility import UserPerspectiveLegibility
from robot_models import *


robot_pos = (1500, 3320, 100)
robot_rot = tr.quaternion_from_euler(ai=math.radians(180), aj=math.radians(0), ak=math.radians(0), axes='rzxy')
robot_pose = Pose(position=robot_pos, orientation=robot_rot)
robot = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='robot',
                                  user_pose=robot_pose, robot_pose=robot_pose)
trajectory = np.array([[1500., 2400., 160.],
                       [1796.115359663, 2474.003143543, 140.454991952],
                       [1928.396107072, 2490.243036578, 131.095203462],
                       [2044.884047189, 2503.013163506, 122.347228354],
                       [2145.632689236, 2511.295613347, 114.390551796],
                       [2230.722447817, 2514.185617642, 107.384708881],
                       [2300.265223349, 2510.893394825, 101.468959406],
                       [2354.410248898, 2500.746463266, 96.761880027],
                       [2393.35168914, 2483.192993484, 93.360773171],
                       [2417.338625372, 2457.806808807, 91.340785283],
                       [2426.688287164, 2424.294705999, 90.753616004],
                       [2421.803747369, 2382.506875336, 91.625680831],
                       [2403.197878222, 2332.451390704, 93.95555612],
                       [2371.526368746, 2274.314095224, 97.710472715],
                       [2327.634472592, 2208.485927902, 102.82149752],
                       [2272.626003983, 2135.601371872, 109.176754041],
                       [2207.972013419, 2056.595867845, 116.611298842],
                       [2135.699871513, 1972.801689797, 124.890216074],
                       [2058.768376073, 1886.134929281, 133.6756471],
                       [1800., 1700., 160.]])

trajectory_2 = np.array([[1500., 2400., 160.],
                    [1619.503525779, 2350.536096482, 288.537908322],
                    [1675.397065052, 2308.749520789, 356.172359955],
                    [1724.772357537, 2268.096166786, 421.037706751],
                    [1768.050482273, 2228.626892636, 480.966165992],
                    [1805.445774542, 2190.339153317, 534.344819237],
                    [1837.029724312, 2153.181836686, 579.983255534],
                    [1862.797522333, 2117.060973664, 616.981926776],
                    [1882.736946321, 2081.847106305, 644.608469443],
                    [1896.893998475, 2047.384681491, 662.190946799],
                    [1905.425926083, 2013.503357706, 669.036866069],
                    [1908.630791616, 1980.030614552, 664.384756953],
                    [1906.944114045, 1946.804646942, 647.391248541],
                    [1900.896988923, 1913.686329053, 617.15186494],
                    [1891.035235264, 1880.569107605, 572.750175286],
                    [1877.803932654, 1847.385962315, 513.331781148],
                    [1861.407569141, 1814.112786453, 438.221098036],
                    [1841.686911482, 1780.766852871, 347.215067953],
                    [1818.341145814, 1747.393509262, 241.969680403],
                    [1800., 1700., 160.]])

robot_trajectory = robot.transformTrajectory(trajectory, viewport=False)
robot_trajectory_2 = robot.transformTrajectory(trajectory_2, viewport=False)

irb = IRB4600()
print(irb)
print(irb.q)
# ur10 = UR10()
# print(ur10)
# ur10_2 = rtb.models.URDF.UR10()
# print(ur10_2)

# UR10e Arm Chain
# armChain = Chain(name='arm', links=[
#     OriginLink(),
#     URDFLink(
#         name="shoulder_pan_joint",
#         bounds=[-6.28318530718, 6.28318530718],
#         translation_vector=[0, 0, 0.181],
#         orientation=[0, 0, 0],
#         rotation=[0, 0, 1],
#     ),
#     URDFLink(
#         name="shoulder_lift_joint",
#         bounds=[-6.28318530718, 6.28318530718],
#         translation_vector=[0, 0.176, 0],
#         orientation=[0, 1.570796, 0],
#         rotation=[0, 1, 0],
#     ),
#     URDFLink(
#         name="elbow_joint",
#         bounds=[-3.14159265359, 3.14159265359],
#         translation_vector=[0, -0.137, 0.613],
#         orientation=[0, 0, 0],
#         rotation=[0, 1, 0],
#     ),
#     URDFLink(
#         name="wrist_1_joint",
#         bounds=[-6.28318530718, 6.28318530718],
#         translation_vector=[0, 0, 0.571],
#         orientation=[0, 1.570796, 0],
#         rotation=[0, 1, 0],
#     ),
#     URDFLink(
#         name="wrist_2_joint",
#         bounds=[-6.28318530718, 6.28318530718],
#         translation_vector=[0, 0.135, 0],
#         orientation=[0, 0, 0],
#         rotation=[0, 0, 1],
#     ),
#     URDFLink(
#         name="wrist_3_joint",
#         bounds=[-6.28318530718, 6.28318530718],
#         translation_vector=[0, 0, 0.12],
#         orientation=[0, 0, 0],
#         rotation=[0, 1, 0],
#     ),
#     URDFLink(
#         name="hand",
#         bounds=[0, 0],
#         translation_vector=[0, 0.295, 0.0],
#         orientation=[0, 0, 0],
#         rotation=[0, 0, 0],
#     )
# ])
#
# ur10_ik_traj = []
# ur10_success = True
irb_ik_traj = []
irb_ik_traj_2 = []
irb_success = True
irb_success_2 = True
for i in range(len(trajectory)):
    x, y, z = robot_trajectory[i]/1000
    x_2, y_2, z_2 = robot_trajectory_2[i] / 1000

    print('-----\nPoint: ' + str(x) + ', ' + str(y) + ', ' + str(z) + '\nAt distance: ' +
          str(np.linalg.norm(robot_trajectory[i])) + '\n---')
    # ur10_ikResults, fail, err = ur10.ikine(sm.SE3(x, y, z) * sm.SE3.OA([0, 1, 0], [0, 0, 1]))
    # ur10_success = ur10_success and (not fail)
    # print('UR10 fail: ' + str(fail) + ' Reason: ' + str(err))

    irb_ikResults, fail, err = irb.ikine(sm.SE3(x, y, z) * sm.SE3.OA([0, 1, 0], [0, 0, 1]))
    irb_success = irb_success and (not fail)
    print('IRB Trajectory 1 fail: ' + str(fail) + ' Reason: ' + str(err))

    print('------\nPoint 2: ' + str(x_2) + ', ' + str(y_2) + ', ' + str(z_2) + '\nAt distance: ' +
          str(np.linalg.norm(robot_trajectory_2[i])) + '\n---')
    irb_ikResults_2, fail, err = irb.ikine(sm.SE3(x_2, y_2, z_2) * sm.SE3.OA([0, 1, 0], [0, 0, 1]))
    irb_success_2 = irb_success_2 and (not fail)
    print('IRB Trajectory 2 fail: ' + str(fail) + ' Reason: ' + str(err))

    # ur10_ik_traj += [ur10_ikResults]
    irb_ik_traj += [irb_ikResults]
    irb_ik_traj_2 += [irb_ikResults_2]
    print('------------------------------------------------\n')

# ur10_ik_traj = np.array(ur10_ik_traj)
irb_ik_traj = np.array(irb_ik_traj)
irb_ik_traj_2 = np.array(irb_ik_traj_2)

ur10_fk_results = []
irb_fk_results = []
for i in range(len(robot_trajectory)):

    # if ur10_success:
    #     fk = ur10.fkine(ur10_ik_traj[i])
    #     ur10_fk_results += [fk.t * 1000]

    if irb_success:
        fk = irb.fkine(irb_ik_traj[i])
        irb_fk_results += [fk.t * 1000]

irb_fk_results = np.array(irb_fk_results)
print(robot_trajectory)
# print(np.array(ur10_fk_results))
print(irb_fk_results)

robot_orientation = Quaternion(robot_pose.orientation[0], robot_pose.orientation[1], 
                              robot_pose.orientation[2], robot_pose.orientation[3])
robot_position = Point(robot_pose.position[0], robot_pose.position[1], robot_pose.position[2])

robot_euler = tr.euler_from_quaternion((robot_orientation.x, robot_orientation.y,
                                      robot_orientation.z, robot_orientation.w))

robot_transformation = tr.euler_matrix(robot_euler[0], robot_euler[1], robot_euler[2])

robot_transformation[0, 3] = robot_position.x
robot_transformation[1, 3] = robot_position.y
robot_transformation[2, 3] = robot_position.z
# robot_transformation = np.linalg.inv(robot_transformation)

irb_world = []
for i in range(len(irb_fk_results)):
    irb_world += [robot_transformation.dot(np.concatenate((irb_fk_results[i], 1), axis=None))[:-1]]

irb_world = np.array(irb_world)
print(trajectory)
print(irb_world)

# print('----- JACOBIANS -----')
# for i in range(len(irb_ik_traj_2)):
#
#     print(irb.q)
#     print(irb.jacob0(irb_ik_traj_2[i]))
#     irb.q = irb_ik_traj[i]
#     print(irb.jacob0(irb_ik_traj_2[i]))
#     irb.q = np.zeros(len(irb.q))
#
#     print('\n')
#     print('------------------------------------')
#
# print(len(robot_trajectory_2) == len(irb_ik_traj_2))
# print(robot_trajectory_2.shape, irb_ik_traj_2.shape)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys
import math
import tf.transformations as T

from image_annotations_3d import ImageAnnotations3D
from geometry_msgs.msg import Pose, Quaternion, Point
from user_perspective_legibility import UserPerspectiveLegibility
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def animate(i, traj, ax):
	# plt.clf()
	ax.plot(traj[:int(i + 1), 0], traj[:int(i + 1), 1], traj[:int(i + 1), 2], 'black', label='Trajectory',
	        markersize=15, marker='.', linestyle="None")
	ax.tick_params(labelsize=17)


def main():

	file_path = os.path.dirname(sys.argv[0])
	full_path = os.path.abspath(file_path)
	image_dir = full_path + '/images'

	user1_rot = T.quaternion_from_euler(ai=math.radians(0), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	user2_rot = T.quaternion_from_euler(ai=math.radians(295), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	user3_rot = T.quaternion_from_euler(ai=math.radians(65), aj=math.radians(99), ak=math.radians(0), axes='rzxy')
	robot_rot = T.quaternion_from_euler(ai=math.radians(180), aj=math.radians(90), ak=math.radians(0), axes='rzxy')

	user1_translation = (1500.0, 1000.0, 100.0)
	user2_translation = (500.0, 1500.0, 100.0)
	user3_translation = (2500.0, 1500.0, 100.0)
	robot_translation = (1500.0, 2500.0, 100.0)

	robot_pose = Pose(position=robot_translation, orientation=robot_rot)
	user1_pose = Pose(position=user1_translation, orientation=user1_rot)
	user2_pose = Pose(position=user2_translation, orientation=user2_rot)
	user3_pose = Pose(position=user3_translation, orientation=user3_rot)

	user1 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user1',
	                                  user_pose=user1_pose, robot_pose=robot_pose)
	user2 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user2',
	                                  user_pose=user2_pose, robot_pose=robot_pose)
	user3 = UserPerspectiveLegibility(using_ros=False, orientation_type='euler', user_id='user3',
	                                  user_pose=user3_pose, robot_pose=robot_pose)

	targets = np.array([[1250.0, 1750.0, 100.0], [1750.0, 1750.0, 100.0]])
	traj = np.array([[1500., 2500., 100.],
	                 [1231.236473867, 2148.075401677, 99.283802767],
	                 [1131.43736223, 1987.656014801, 101.159531471],
	                 [1061.634360087, 1853.386732753, 102.751306656],
	                 [1018.512789725, 1746.313538961, 103.354038468],
	                 [994.391937503, 1666.48981711, 103.056748249],
	                 [980.256930737, 1612.559974547, 102.413168293],
	                 [970.315794025, 1582.084418324, 101.92195763],
	                 [963.485059859, 1571.626476946, 101.789114254],
	                 [960.89816097, 1576.546303414, 101.969438592],
	                 [962.90390306, 1591.570379823, 102.316582708],
	                 [968.36633932, 1612.007032007, 102.698731155],
	                 [975.652273191, 1634.630351748, 103.030157699],
	                 [983.591544785, 1657.879944972, 103.255690999],
	                 [991.744752302, 1681.635638644, 103.351599215],
	                 [1000.104029019, 1706.879637201, 103.413823695],
	                 [1008.49003296, 1735.383432178, 103.902013841],
	                 [1015.943798163, 1769.298098667, 106.016031861],
	                 [1020.778283925, 1809.959893652, 111.702234481],
	                 [1250., 1750., 100.]])

	cups = np.array([[
			[1200.0, 1700.0, 115.0], [1300.0, 1700.0, 115.0], [1270.0, 1725.0, 85.0], [1230.0, 1725.0, 85.0],
			[1200.0, 1700.0, 115.0], [1200.0, 1800.0, 115.0], [1230.0, 1775.0, 85.0], [1230.0, 1725.0, 85.0],
			[1200.0, 1700.0, 115.0], [1200.0, 1800.0, 115.0], [1300.0, 1800.0, 115.0], [1270.0, 1775.0, 85.0],
			[1230.0, 1775.0, 85.0], [1200.0, 1800.0, 115.0], [1300.0, 1800.0, 115.0], [1300.0, 1700.0, 115.0],
			[1270.0, 1725.0, 85.0], [1270.0, 1775.0, 85.0], [1300.0, 1800.0, 115.0], [1300.0, 1700.0, 115.0],
			[1270.0, 1725.0, 85.0], [1230.0, 1725.0, 85.0], [1230.0, 1775.0, 85.0], [1270.0, 1775.0, 85.0],
			[1270.0, 1725.0, 85.0]
		],
		[
			[1700.0, 1700.0, 115.0], [1800.0, 1700.0, 115.0], [1770.0, 1725.0, 85.0], [1730.0, 1725.0, 85.0],
			[1700.0, 1700.0, 115.0], [1700.0, 1800.0, 115.0], [1730.0, 1775.0, 85.0], [1730.0, 1725.0, 85.0],
			[1700.0, 1700.0, 115.0], [1700.0, 1800.0, 115.0], [1800.0, 1800.0, 115.0], [1770.0, 1775.0, 85.0],
			[1730.0, 1775.0, 85.0], [1700.0, 1800.0, 115.0], [1800.0, 1800.0, 115.0], [1800.0, 1700.0, 115.0],
			[1770.0, 1725.0, 85.0], [1770.0, 1775.0, 85.0], [1770.0, 1775.0, 85.0], [1800.0, 1800.0, 115.0],
			[1770.0, 1775.0, 85.0], [1730.0, 1775.0, 85.0], [1730.0, 1725.0, 85.0], [1770.0, 1725.0, 85.0],
			[1770.0, 1775.0, 85.0]
		]])

	table = np.array([
		[1900.0, 1650.0, 84.0], [1900.0, 1650.0, 74.0], [1100.0, 1650.0, 74.0], [1100.0, 1650.0, 84.0],
		[1900.0, 1650.0, 84.0], [1900.0, 1850.0, 84.0], [1900.0, 1850.0, 74.0], [1900.0, 1650.0, 74.0],
		[1900.0, 1650.0, 84.0], [1900.0, 1850.0, 84.0], [1900.0, 1850.0, 74.0], [1100.0, 1850.0, 74.0],
		[1100.0, 1850.0, 84.0], [1900.0, 1850.0, 84.0], [1900.0, 1850.0, 74.0], [1100.0, 1850.0, 74.0],
		[1100.0, 1650.0, 74.0], [1100.0, 1650.0, 84.0], [1100.0, 1850.0, 84.0], [1100.0, 1850.0, 74.0],
		[1100.0, 1650.0, 74.0], [1900.0, 1650.0, 74.0], [1900.0, 1850.0, 74.0], [1100.0, 1850.0, 74.0],
		[1100.0, 1650.0, 74.0]
	])

	cups_shape = cups.shape
	table_shape = table.shape

	user1_traj = user1.transformTrajectory(traj, viewport=False)
	user2_traj = user2.transformTrajectory(traj, viewport=False)
	user3_traj = user3.transformTrajectory(traj, viewport=False)

	user1_targets = user1.transformTrajectory(targets, viewport=False)
	user1_robot = user1.transformTrajectory(np.array([robot_translation]), viewport=False)
	u1_transform_matrix, _ = user1.getTransformationMatrices()
	u1_transform_matrix = np.delete(np.delete(u1_transform_matrix, -1, 0), -1, 1)
	u1_traj_world = np.linalg.inv(u1_transform_matrix).dot(user1_traj.T).T
	u1_targets_world = np.linalg.inv(u1_transform_matrix).dot(user1_targets.T).T
	
	user2_targets = user2.transformTrajectory(targets, viewport=False)
	user2_robot = user2.transformTrajectory(np.array([robot_translation]), viewport=False)
	u2_transform_matrix, _ = user3.getTransformationMatrices()
	u2_transform_matrix = np.delete(np.delete(u2_transform_matrix, -1, 0), -1, 1)
	u2_traj_world = np.linalg.inv(u2_transform_matrix).dot(user2_traj.T).T
	u2_targets_world = np.linalg.inv(u2_transform_matrix).dot(user2_targets.T).T
	
	u3_transform_matrix, _ = user3.getTransformationMatrices()
	u3_transform_matrix = np.delete(np.delete(u3_transform_matrix, -1, 0), -1, 1)
	user3_robot = user3.transformTrajectory(np.array([robot_translation]), viewport=False)
	u3_traj_world = np.linalg.inv(u3_transform_matrix).dot(user3_traj.T).T
	user3_targets = user3.transformTrajectory(targets, viewport=False)
	u3_targets_world = np.linalg.inv(u3_transform_matrix).dot(user3_targets.T).T
	user3_cup_1 = user3.transformTrajectory(cups[0], viewport=False)
	user3_cup_1 = np.linalg.inv(u3_transform_matrix).dot(user3_cup_1.T).T
	user3_cup_2 = user3.transformTrajectory(cups[1], viewport=False)
	user3_cup_2 = np.linalg.inv(u3_transform_matrix).dot(user3_cup_2.T).T
	user3_table = user3.transformTrajectory(table, viewport=False)
	user3_table = np.linalg.inv(u3_transform_matrix).dot(user3_table.T).T

	verts_cup_1 = [[user3_cup_1[i, 0], user3_cup_1[i, 1], user3_cup_1[i, 2]] for i in range(cups_shape[1])]
	verts_cup_2 = [[user3_cup_2[i, 0], user3_cup_2[i, 1], user3_cup_2[i, 2]] for i in range(cups_shape[1])]
	verts_table = [[user3_table[i, 0], user3_table[i, 1], user3_table[i, 2]] for i in range(table_shape[0])]
	traj_max = np.max(u3_traj_world, axis=0)
	traj_min = np.min(u3_traj_world, axis=0)
	fig = plt.figure(figsize=(25, 25))
	ax = fig.gca(projection='3d')
	ax.plot(np.array([u3_targets_world[0, 0]]), np.array([u3_targets_world[0, 1]]), np.array([u3_targets_world[0, 2]]),
	        color='darkorange', marker='D', markersize=5)
	ax.plot(np.array([u3_targets_world[1, 0]]), np.array([u3_targets_world[1, 1]]), np.array([u3_targets_world[1, 2]]),
	        color='darkorange', marker='D', markersize=5)
	ax.plot(user3_cup_1[:, 0], user3_cup_1[:, 1], user3_cup_1[:, 2], color='black')
	ax.plot(user3_cup_2[:, 0], user3_cup_2[:, 1], user3_cup_2[:, 2], color='blue')
	ax.plot(user3_table[:, 0], user3_table[:, 1], user3_table[:, 2], color='brown')
	ax.add_collection3d(Poly3DCollection([verts_table[0:5]], color='brown', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_table[5:10]], color='brown', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_table[10:15]], color='brown', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_table[15:20]], color='brown', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_table[20:25]], color='brown', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_1[0:5]], color='black', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_1[5:10]], color='black', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_1[10:15]], color='black', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_1[15:20]], color='black', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_1[20:25]], color='black', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_2[0:5]], color='blue', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_2[5:10]], color='blue', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_2[10:15]], color='blue', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_2[15:20]], color='blue', zsort='max'))
	ax.add_collection3d(Poly3DCollection([verts_cup_2[20:25]], color='blue', zsort='max'))
	ax.axis('off')
	# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	plt.tight_layout()
	# ax.plot(user3_traj[:, 0], user3_traj[:, 1], user3_traj[:, 2], 'black', label='Trajectory', markersize=15, marker='.',
	#         linestyle="None")
	# ax.plot(np.array([robot_translation[0]]), np.array([robot_translation[1]]), np.array([robot_translation[2]]),
	#         color='blue', marker='2', markersize=15, label='Robot')
	# plt.legend(loc='best')
	# ax2 = fig.add_subplot(111, frame_on=False)
	# ax2.axis('off')
	# ax2.axis([0, 1, 0, 1])
	ax.set_xlim3d(traj_min[0]-100, traj_max[0]+500)
	ax.set_ylim3d(traj_min[1]-100, traj_max[1]+500)
	ax.set_zlim3d(traj_min[2]-100, traj_max[2]+100)
	# x_min, x_max = ax.get_xlim3d()
	# y_min, y_max = ax.get_ylim3d()
	# z_min, z_max = ax.get_zlim3d()
	# plt.xticks(np.arange(x_min, x_max, 100))
	# plt.yticks(np.arange(y_min, y_max, 100))
	# ax.set_zticks((np.arange(z_min, z_max, 10)))
	ax.view_init(azim=-35, elev=9)
	# fig.show()
	# plt.pause(5.0)

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
	ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(traj), repeat=True, fargs=(u3_traj_world, ax))
	fig.show()
	input()


if __name__ == '__main__':
	main()

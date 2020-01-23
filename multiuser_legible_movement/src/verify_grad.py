#! /usr/bin/env python2
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian


def C_mat(x):

	input_len, input_dim = x.shape
	if input_len > 2:
		vel_matrix = np.zeros((input_len - 1, input_len - 2))
		vel_matrix[0, 0] = 1
		vel_matrix[-1, -1] = -1
		for i in range(1, input_len - 2):
			vel_matrix[i, i - 1] = -1
			vel_matrix[i, i] = 1
		e = np.array(list([-x[0, :]]) + [[0, 0, 0]]*(input_len - 3) + list([x[-1, :]]))

		k = np.kron(vel_matrix, np.eye(input_dim))
		A = np.dot(k.T, k)
		b = np.dot(k.T, e.ravel())
		c = np.dot(e.ravel().T, e.ravel()) / 2

		A_term = np.dot(np.dot(x[1:-1].ravel().T, A), x[1:-1].ravel())
		b_term = np.dot(x[1:-1].ravel().T, b)
		cost_prev = float(1 / 2.0) * A_term + b_term + c  # * 1000

	else:
		if input_len > 1:
			vel_matrix = -np.ones(input_len - 1)
			e = np.array(list([x[-1, :]]))

			k = np.kron(vel_matrix, np.eye(input_dim))
			A = np.dot(k.T, k)
			b = np.dot(k.T, e.ravel())
			c = np.dot(e.ravel().T, e.ravel()) / 2

			A_term = np.dot(np.dot(x[0].ravel().T, A), x[0].ravel())
			b_term = np.dot(x[0].ravel().T, b)

		else:
			vel_matrix = np.zeros(1)
			e = -x[:input_len + 1] * 0

			k = np.kron(vel_matrix, np.eye(input_dim))
			A = np.dot(k.T, k)
			b = np.dot(k.T, e.ravel())
			c = np.dot(e.ravel().T, e.ravel()) / 2

			A_term = np.dot(np.dot(x.ravel().T, A), x.ravel())
			b_term = np.dot(x.ravel().T, b)

		cost_prev = float(1 / 2.0) * A_term + b_term + c  # * 1000

	return cost_prev


def C_grad(x):

	input_len, input_dim = x.shape
	if input_len > 2:
		vel_matrix = np.zeros((input_len - 1, input_len - 2))
		vel_matrix[0, 0] = 1
		vel_matrix[-1, -1] = -1
		for i in range(1, input_len - 2):
			vel_matrix[i, i - 1] = -1
			vel_matrix[i, i] = 1
		e = np.array(list([-x[0, :]]) + [[0, 0, 0]] * (input_len - 3) + list([x[-1, :]]))

		k = np.kron(vel_matrix, np.eye(input_dim))
		A = np.dot(k.T, k)
		b = np.dot(k.T, e.ravel())

		A_term = np.dot(A.T, x[1:-1].ravel())
		b_term = b
		grad_cost = float(1 / 2.0) * A_term + b_term # * 1000

	else:
		if input_len > 1:
			vel_matrix = -np.ones(input_len - 1)
			e = np.array(list([x[-1, :]]))

			k = np.kron(vel_matrix, np.eye(input_dim))
			A = np.dot(k.T, k)
			b = np.dot(k.T, e.ravel())

			A_term = np.dot(A.T, x[0].ravel())
			b_term = b

		else:
			vel_matrix = np.zeros(1)
			e = -x[:input_len + 1] * 0

			k = np.kron(vel_matrix, np.eye(input_dim))
			A = np.dot(k.T, k)
			b = np.dot(k.T, e.ravel())

			A_term = np.dot(A.T, x.ravel())
			b_term = b

		grad_cost = A_term + b_term # * 1000

	return grad_cost


def C_simple(x):

	traj_len, traj_dim = x.shape
	cost = np.zeros(1)

	for i in range(traj_len-1):

		next_point = x[i + 1]
		point = x[i]

		cost = cost + (np.sum(next_point ** 2) + np.sum(point ** 2) - 2 * np.sum(next_point * point))

	cost = cost/2

	return cost


def main():

	D_c = grad(C_mat)
	# D_c_simple = grad(C_simple)

	# targets = np.array([[1.35, 1.45, 1.25], [1.35, 1.35, 1.25]])
	targets = np.array([[350, 450, 250], [350, 350, 250]])
	# robot_translation = (1.70, 1.40, 0.0)
	robot_translation = (700, 400, 0.0)
	traj_x = np.linspace(robot_translation[0], targets[0, 0], num=10)[:, None]
	traj_y = np.linspace(robot_translation[1], targets[0, 1], num=10)[:, None]
	traj_z = np.linspace(targets[0, 2], targets[0, 2], num=10)[:, None]
	traj = np.hstack((traj_x, traj_y, traj_z))
	traj = np.array([[1, 2, 3], [1.5, 2.7, 3], [2, 3.4, 3], [2.5, 4.1, 3], [3, 4.8, 3]])
	# traj_reverse = traj[::-1, :]

	for i in range(len(traj)):
		traj_grad = map(D_c, np.array([traj[:i+1]]))[0]
		# traj_grad = map(D_c_simple, np.array([traj[:i+1]]))[0]
		# traj_grad_reverse = map(D_c, np.array([traj_reverse[:i+1]]))[0]
		traj_grad_forward = map(D_c, np.array([traj[i:]]))[0]
		# traj_grad_forward = map(D_c_simple, np.array([traj[i:]]))[0]
		traj_grad_forward_hand = C_grad(traj[i:])
		print('-----------------------')
		print(len(traj[:i+1]))
		print(C_mat(traj[:i + 1]))
		print(np.sum(traj_grad))
		print(C_simple(traj[:i + 1])[0])
		print(np.sum(C_grad(traj[:i+1])))
		print('-----------------------')
		print(len(traj[i:]))
		print(C_mat(traj[i:]))
		# print(traj_grad_forward)
		# print(traj_grad_forward_hand)
		print(np.sum(traj_grad_forward))
		print(C_simple(traj[i:])[0])
		print(np.sum(traj_grad_forward_hand))
		print('----------------------\n')

		# fig = plt.figure(1)
		# plt.clf()
		# plt.plot(traj[i:, 0], traj_grad_forward[:, 0])
		# fig.show()

		# fig = plt.figure(2)
		# plt.clf()
		# plt.plot(traj[i:, 1], traj_grad_forward[:, 1])
		# fig.show()

		# fig = plt.figure(3)
		# plt.clf()
		# plt.plot(traj[i:, 2], traj_grad_forward[:, 2])
		# fig.show()

		x = raw_input()


if __name__ == '__main__':
	main()

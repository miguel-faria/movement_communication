# ROS message for a PMP trajectory. each entry of traj and time_step must have the same length
PMPPoint[] 	traj		# vector, of the same size as the robot's DOFs, with the sequence of joint values at each time step for each DOF.
float64[] 	time_step	# times of observations starting at zero
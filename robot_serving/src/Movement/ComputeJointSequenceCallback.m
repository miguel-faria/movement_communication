function resp = ComputeJointSequenceCallback( ~, req, resp, robot_pmp)
%ComputeJointSequenceCallback ROS callback to the service for obtaining
%robot movement
%   resp = ComputeJointSequenceCallback( ~, req, resp, robot_pmp) produces the
%   sequence of joint angles, using ProMPs, for a robot movement that has
%   to finish at the point given by req.
%   This function is meant to be used as a callback for a ROS service,
%   which receives the (X,Y,Z) position to finish the robot's arm movement
%   and returns the sequence of joint angles to perform the movement that
%   finishes as close as possible to the given position.

assert(strcmp(class(robot_pmp), 'OriginalProMP') == 1, '"robot_pmp" must be a ProMP.');

disp('Get Pos Service Callback');

finish_coords = [req.XPos; req.YPos];%; req.ZPos];

conditioned_pmp = robot_pmp.conditionNonDestructive(finish_coords);

probable_trajectory = conditioned_pmp.mostProbable();

resp.RobotTrajectory = robotics.ros.custom.msggen.robot_serving.PMPTraj;
resp.RobotTrajectory.TimeStep = probable_trajectory(:,1);
resp.RobotTrajectory.Traj = robotics.ros.custom.msggen.robot_serving.PMPPoint;
points = robotics.ros.custom.msggen.robot_serving.PMPPoint;

for i = 2:size(probable_trajectory, 2)
    points.JointAngles = probable_trajectory(:,i);
    resp.RobotTrajectory.Traj(i-1,1) = points;
end

end


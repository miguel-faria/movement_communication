close all;
clear;

%Setting up service environment
%predictable_train_data = csvread('train_data_predictable.txt');
%legible_train_data = csvread('train_data_legible.txt'); % data must be  
                                        % ordered by time and 
                                        % at each time step the data must 
                                        % follow demo1 dof1 | demo1 dof2 
                                        % | ... | demoN dof1 | demoN dof2 |
predictable_train_data = csvread('train_data.txt');
legible_train_data = csvread('train_data.txt');
                                        
                                        
dof = 2;
dt = 0.01;
sigma_y = 0.0001*eye(dof);

predictable_robot_pmp = OriginalProMP(predictable_train_data, dof, dt, sigma_y);
legible_robot_pmp = OriginalProMP(legible_train_data, dof, dt, sigma_y);

predictable_robot_pmp.build(LinearPhaseGenerator(), NormalizedGaussianBasisGenerator(10), false);
legible_robot_pmp.build(LinearPhaseGenerator(), NormalizedGaussianBasisGenerator(10), false);      
res = robotics.ros.custom.msggen.robot_serving.MovementResponse;

%Connecting to ROS
disp('Connecting to ROS');
%rosinit('http://localhost:11311');
%masterHost = 'localhost';
rosinit('http://bea.local:11311');
masterHost = 'bea.local';
%node = robotics.ros.Node('/movement_decision', masterHost);

%Startup Movement Computation Service
disp('Creating Mixed (Legible/Predictable) Movement Decision server');
legible_rob_server = rossvcserver('/movement_decision_legible',...
    robotics.ros.custom.msggen.CustomMsgConsts.robot_serving_Movement,  {@ComputeJointSequenceCallback legible_robot_pmp});
predictable_rob_server = rossvcserver('/movement_decision_predictable',...
    robotics.ros.custom.msggen.CustomMsgConsts.robot_serving_Movement,  {@ComputeJointSequenceCallback predictable_robot_pmp});

str = input('Enter any key to terminate\n','s');

%Disconnecting from ROS
%clear('mov_rob_server');
%clear('node');
%clear('masterHost');

rosshutdown;
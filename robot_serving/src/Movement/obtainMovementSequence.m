close all;
clear;

%Setting up service environment
train_data = csvread('train_data.txt'); % data must be ordered by time and 
                                        % at each time step the data must 
                                        % follow demo1 dof1 | demo1 dof2 
                                        % | ... | demoN dof1 | demoN dof2 |

dof = 2;
dt = 0.01;
sigma_y = 0.0001*eye(dof);

robot_pmp = OriginalProMP(train_data, dof, dt, sigma_y);
robot_pmp.build(LinearPhaseGenerator(), NormalizedGaussianBasisGenerator(10), false);      
res = robotics.ros.custom.msggen.robot_serving.MovementResponse;

%Connecting to ROS
disp('Connecting to ROS');
rosinit('http://localhost:11311');
masterHost = 'localhost';
%node = robotics.ros.Node('/movement_decision', masterHost);

%Startup Movement Computation Service
disp('Creating service server');
mov_rob_server = rossvcserver('/movement_decision',...
    robotics.ros.custom.msggen.CustomMsgConsts.robot_serving_Movement,  {@ComputeJointSequenceCallback robot_pmp});
    
% mov_rob_server = robotics.ros.ServiceServer(node, '/movement_decision',...
%     rostype.roscpp_tutorials_TwoInts,  @exampleHelperROSSumCallback);
str = input('Enter any key to terminate\n','s');

%Disconnecting from ROS
%clear('mov_rob_server');
%clear('node');
%clear('masterHost');

rosshutdown;
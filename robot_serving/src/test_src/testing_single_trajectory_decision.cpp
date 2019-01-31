/*
 * trajectory_decision.cpp
 *
 *  Created on: Jun 17, 2016
 *      Author: miguel
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/service_client.h>
#include <cmath>
#include <iostream>
#include <map>
#include <string>

#include "../../devel/include/robot_serving/Movement.h"
#include "../Movement/Trajectory_Types/Single_Trajectory.h"

/**
 *
 * Application to testing if the mixed trajectory
 * decision is done correctly. Each call to the service
 * will test one specific case and the result will be printed.
 *
 */

using namespace std;
using namespace cv;
using namespace ros;
using namespace robot_serving;
using namespace movement_decision;

double euclidean_distance(Point3f ptx1, Point3f ptx2){

	return sqrt(pow(ptx1.x - ptx2.x, 2) + pow(ptx1.y - ptx2.y, 2) + pow(ptx1.z - ptx2.z, 2));
}

int main(int argc, char **argv) {

	string legible_uri = "/legible_traj";
	string predictable_uri = "/predictable_traj";
	ServiceClient result;
	map<string, Point3f> cups1, cups2;
	map<string, double> cups1_dist, cups2_dist;
	Point3f point_zero = Point3f(0, 0, 0);

	init(argc, argv, "mixed_trajectory_tester");
	NodeHandle nh;
	Single_Trajectory* pt = new Single_Trajectory();
	Single_Trajectory* lt = new Single_Trajectory();
	lt->set_new_traj_srv(nh.serviceClient<robot_serving::Movement>(legible_uri));
	pt->set_new_traj_srv(nh.serviceClient<robot_serving::Movement>(predictable_uri));

	cout << "------------------" << endl << "----- TEST 1 -----" << endl << "------------------" << endl;
	cout << "Legible" << endl;

	result = lt->choose_trajectory_service(cups1, cups1_dist, "red", "right");

	result.getService() == legible_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << result.getService() << endl;
	cout << "------------------" << endl << "----- TEST 2 -----" << endl << "------------------" << endl;
	cout << "Predictable" << endl;

	result = pt->choose_trajectory_service(cups2, cups2_dist, "red", "right");

	result.getService() == predictable_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << result.getService() << endl;

	return 0;

}


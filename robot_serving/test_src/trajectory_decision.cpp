/*
 * trajectory_decision.cpp
 *
 *  Created on: Jun 17, 2016
 *      Author: miguel
 */

#include <opencv2/core/core.hpp>
#include <ros/node_handle.h>
#include <ros/service_client.h>
#include <iostream>
#include <map>
#include <string>

#include "../devel/include/robot_serving/Movement.h"
#include "../src/Movement/Trajectory_Types/Mixed_Trajectory.h"

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
	Mixed_Trajectory mt = Mixed_Trajectory();
	mt.set_legible_trajectory_srv(nh.serviceClient<robot_serving::Movement>(legible_uri));
	mt.set_predictable_trajectory_srv(nh.serviceClient<robot_serving::Movement>(predictable_uri));

	//Cups1 maps initialization
	cups1.insert(pair<string, Point3f>("red", Point3f(-500, 250, 1000)));
	cups1.insert(pair<string, Point3f>("green", Point3f(-200, 250, 1000)));
	cups1.insert(pair<string, Point3f>("blue", Point3f(150, 250, 1000)));
	cups1_dist.insert(pair<string, double>("red", euclidean_distance(cups1["red"], point_zero)));
	cups1_dist.insert(pair<string, double>("blue", euclidean_distance(cups1["blue"], point_zero)));
	cups1_dist.insert(pair<string, double>("green", euclidean_distance(cups1["green"], point_zero)));

	//Cups2 maps initialization
	cups2.insert(pair<string, Point3f>("red", Point3f(-600, 250, 1000)));
	cups2.insert(pair<string, Point3f>("green", Point3f(-450, 250, 1000)));
	cups2.insert(pair<string, Point3f>("blue", Point3f(100, 250, 1000)));
	cups2.insert(pair<string, Point3f>("pink", Point3f(-150, 250, 1000)));
	cups2_dist.insert(pair<string, double>("red", euclidean_distance(cups2["red"], point_zero)));
	cups2_dist.insert(pair<string, double>("blue", euclidean_distance(cups2["blue"], point_zero)));
	cups2_dist.insert(pair<string, double>("green", euclidean_distance(cups2["green"], point_zero)));
	cups2_dist.insert(pair<string, double>("pink", euclidean_distance(cups2["pink"], point_zero)));


	cout << "------------------" << endl << "----- TEST 1 -----" << endl << "------------------" << endl;
	cout << "Last cup on the right! - Legible" << endl;

	result = mt.choose_trajectory_service(cups1, cups1_dist, "red", "right");

	result.getService() == legible_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << "------------------" << endl << "----- TEST 2 -----" << endl << "------------------" << endl;
	cout << "Green Cup with space! - Legible" << endl;

	result = mt.choose_trajectory_service(cups1, cups1_dist, "green", "right");

	result.getService() == legible_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << "------------------" << endl << "----- TEST 3 -----" << endl << "------------------" << endl;
	cout << "Cups equally spaced! - Legible" << endl;

	cups1_dist["red"] = cups1_dist["blue"] = cups1_dist["green"] = 1000;
	result = mt.choose_trajectory_service(cups1, cups1_dist, "red", "right");

	result.getService() == legible_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << "------------------" << endl << "----- TEST 4 -----" << endl << "------------------" << endl;
	cout << "Green Cup surrounded! - Predictable" << endl;

	result = mt.choose_trajectory_service(cups2, cups2_dist, "green", "right");

	result.getService() == predictable_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << "------------------" << endl << "----- TEST 5 -----" << endl << "------------------" << endl;
	cout << "Cup close to left one! - Predictable" << endl;

	result = mt.choose_trajectory_service(cups2, cups2_dist, "pink", "left");

	result.getService() == predictable_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;
	cout << "------------------" << endl << "----- TEST 6 -----" << endl << "------------------" << endl;
	cout << "Last Cup to serve! - Predictable" << endl;

	cups2.erase("blue");
	cups2.erase("green");
	cups2.erase("pink");
	result = mt.choose_trajectory_service(cups2, cups2_dist, "red", "right");

	result.getService() == predictable_uri ? cout << "PASSED!!!" << endl : cout << "FAILED!!!" << endl;


	return 0;

}


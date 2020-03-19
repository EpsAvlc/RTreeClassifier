#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_box.h>
#include <thread>
#include <chrono>

#include "kitti_helper.h"


using namespace std;
using namespace pcl;
int main(int argc, char** argv)
{
	visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    KITTIHelper kh;
    string cloudfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/point_cloud/000156.bin";
    
    string labelfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/label/000156.txt";
    
    string calibfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/calib/000156.txt";

	vector<PointCloud<PointXYZI>::Ptr> clusters;
	kh.ExtractClusters(cloudfile, labelfile, calibfile, clusters);

	for(int i = 0; i < clusters.size(); i++)
	{
		cout << clusters[i]->size() << endl;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> 
			cluster_color(clusters[i], 255, 0, 0);
    	viewer->addPointCloud<pcl::PointXYZI> (clusters[i], cluster_color, "cluster_" + to_string(i));
	}

    while(!viewer->wasStopped())
    {
        viewer->spinOnce (100);
    	std::this_thread::sleep_for(chrono::milliseconds(30));
    }
    return 0;
}
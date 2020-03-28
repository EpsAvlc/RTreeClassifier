#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_box.h>

#include <opencv2/core.hpp>

#include <thread>
#include <chrono>

#include "kitti_helper.h"
#include "EVDescriptor.h"
#include "RForest.h"
#include "common.h"

using namespace std;
using namespace pcl;
using namespace cv;

int main(int argc, char** argv)
{
    FileStorage fs("/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/demo/config/demo_predict.yaml", FileStorage::READ);

    int cloud_index;
    fs["cloud_index"] >> cloud_index;
    char cloud_index_str[6];
    sprintf(cloud_index_str, "%06d", cloud_index);

    string cloudfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/point_cloud/"+ string(cloud_index_str) + ".bin";
    
    string labelfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/label/" + string(cloud_index_str) + ".txt";
    
    string calibfile= "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/calib/" + string(cloud_index_str) + ".txt";

    KITTIHelper kh;

    PointCloud<PointXYZI>::Ptr input_cloud;
    kh.ReadPointCloud(cloudfile, input_cloud);

    vector<PointCloud<PointXYZI>::Ptr> clusters;
    vector<uint> labels;
    kh.ExtractClusters(cloudfile, labelfile, calibfile, clusters, labels, 20);

	visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    viewer->addPointCloud<PointXYZI>(input_cloud, "input_cloud");

    RForest rf("/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/model/kitti.xml");

    for(int i = 0; i < clusters.size(); i++)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> 
			cluster_color(clusters[i], 0, 255, 0);
        viewer->addPointCloud<PointXYZI>(clusters[i], cluster_color, "cluster_" + to_string(i));
        Eigen::VectorXf des;
        EVDescriptor::ExtractDescriptor(clusters[i], des);
        float prob = rf.Predict(des);
        int label = (int)round(prob);
        string label_str;
        if(label == 0)
            label_str = "Car";
        else if(label == 1)
            label_str = "Pedestrian";
        else
            label_str = "Others";
        PointXYZI centroid;
        computeCentroid(*clusters[i], centroid);
        viewer->addText3D(label_str, centroid, 0.5, 1.0, 0, 0, "text_" + to_string(i));
        // cout << label << endl;
        // cout << prob << endl;
    }




    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    return 0;
}
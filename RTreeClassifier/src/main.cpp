#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_box.h>
#include <thread>
#include <chrono>

#include "kitti_helper.h"
#include "EVDescriptor.h"
#include "RForest.h"
#include "common.h"

using namespace std;
using namespace pcl;

void GetDescriptorsAndLabels(int start, int end, 
    Eigen::MatrixXf& descriptor_matrix,  Eigen::MatrixXf& label_matrix)
{
    KITTIHelper kh;
    string cloudfile_path = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/point_cloud/";
    
    string labelfile_path = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/label/";
    
    string calibfile_path = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/calib/";

    vector<Eigen::VectorXf> all_descriptors;
    vector<int> all_labels;
    #pragma omp parallel for
    for(int i = start; i < end; i++)
    {
        char index[6];
        sprintf(index, "%06d", i);
        string cloudfile = cloudfile_path + string(index) + ".bin";
        string labelfile = labelfile_path + string(index) + ".txt";
        string calibfile = calibfile_path + string(index) + ".txt";
        vector<PointCloud<PointXYZI>::Ptr> clusters;
        vector<uint> labels;
        kh.ExtractClusters(cloudfile, labelfile, calibfile, 
            clusters, labels, 20);
        vector<Eigen::VectorXf> descriptors;
        for(int j = 0; j < clusters.size(); j++)
        {
            Eigen::VectorXf des;
            EVDescriptor::ExtractDescriptor(clusters[j], des);
            descriptors.push_back(des);
        }

        all_descriptors.insert(all_descriptors.end(), descriptors.begin(), descriptors.end());
        all_labels.insert(all_labels.end(), labels.begin(), labels.end());
        LOG(INFO) << "Complete the " << i << " th file.";
    }

    CHECK_EQ(all_descriptors.size(), all_labels.size());

    descriptor_matrix = Eigen::MatrixXf::Zero(7, all_descriptors.size());
    for(int i = 0; i < all_descriptors.size(); i++)
    {
        descriptor_matrix.block(0, i, 7, 1) = all_descriptors[i];
    }
    descriptor_matrix.transposeInPlace();

    label_matrix = Eigen::MatrixXf::Zero(all_labels.size(), 1);
    for(int i = 0; i < all_labels.size(); i++)
    {
        if(all_labels[i] == 1)
            label_matrix(i, 0) = all_labels[i];
        else
            label_matrix(i, 0) = 0;
    }
}

int main(int argc, char** argv)
{
	// visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // viewer->setBackgroundColor(0, 0, 0);
    // viewer->addCoordinateSystem(1.0);
    // viewer->initCameraParameters();

   
    RForest rf;
    Eigen::MatrixXf descriptor_matrix;
    Eigen::MatrixXf label_matrix;
    GetDescriptorsAndLabels(0, 800, descriptor_matrix, label_matrix);
    rf.Train(descriptor_matrix, label_matrix);
    GetDescriptorsAndLabels(801, 1000, descriptor_matrix, label_matrix);
    rf.Test(descriptor_matrix, label_matrix);
    return 0;
}
#include <iostream>
#include <pcl/visualization/cloud_viewer.h>

#include "kitti_helper.h"

using namespace std;
using namespace pcl;
int main(int argc, char** argv)
{
    KITTIHelper kh;
    string infile = "/home/cm/Workspaces/RtreeClassifier/small_training_set/point_cloud/000000.bin";
    
    PointCloud<PointXYZI>::Ptr out_cloud(new PointCloud<PointXYZI>);
    kh.ReadPointCloud(infile, out_cloud); 
    
    string labelfile = "/home/cm/Workspaces/RtreeClassifier/small_training_set/label/000000.txt";
    
    vector<KITTIHelper::BBox> bboxes;
    kh.ParseLabel(labelfile, bboxes);

    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(out_cloud);
    while(!viewer.wasStopped())
    {
    
    }
    return 0;
}
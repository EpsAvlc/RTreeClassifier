/*
 * Created on Mon Mar 16 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

class KITTIHelper
{
public:
    struct BBox
    {
       int type = -1; 
       float height;
       float width;
       float length;
       float x, y, z;
       float rot; 
    };
    void ReadPointCloud(const std::string& infile, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_ptr);
    void ParseLabel(const std::string& infile, std::vector<BBox>& bboxes);
    void ParseCalib(const std::string& infile, Eigen::MatrixXf& velo_to_cam, Eigen::Matrix3f& R0_rect);
    void ExtractClusters(const std::string& cloudfile, 
        const std::string& labelfile, const std::string& calibfile,
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters,
        std::vector<uint>& labels, int size_thres);
    void RemoveGround(pcl::PointCloud<pcl::PointXYZI>::Ptr ori_cloud, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud);

private:
    void split(std::string& s, char delim, std::vector<std::string>& strs);

};
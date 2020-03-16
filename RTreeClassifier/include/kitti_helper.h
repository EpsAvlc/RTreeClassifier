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

private:
    void split(std::string& s, char delim, std::vector<std::string>& strs);

};
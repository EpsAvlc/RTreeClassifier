/*
 * Created on Fri Mar 20 2020
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

/**
 * @brief Eigen value based describetor.
 * 
 */
class EVDescriptor
{
public:
    void ExtractDescriptor(const pcl::PointCloud<pcl::PointXYZI>::Ptr cluster, 
        Eigen::VectorXf& ev_descriptor);
};
/*
 * Created on Fri Mar 20 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#ifndef EVDESCRIPTRO_H__
#define EVDESCRIPTOR_H__
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
/**
 * @brief Extract descriptors from clusters.
 * 
 * @param cluster input point cloud clusters.
 * @param ev_descriptor output descriptors. See SegMatch.
 */
    static void ExtractDescriptor(const pcl::PointCloud<pcl::PointXYZI>::Ptr cluster, 
        Eigen::VectorXf& ev_descriptor);
};
#endif // !EVDESCRIPTRO_H__

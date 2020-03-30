/*
 * Created on Mon Mar 30 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#ifndef ESFDESCRIPTOR_H__
#define ESFDESCRIPTOR_H__

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <pcl/features/esf.h>

class ESFDescriptor
{
public:
    void ExtractDescriptor(const pcl::PointCloud<pcl::PointXYZI>::Ptr cluster, 
        Eigen::VectorXf& ev_descriptor);
private:
    pcl::ESFEstimation<pcl::PointXYZI, pcl::ESFSignature640> esf_estimator_;
};
#endif
/*
 * Created on Mon Mar 30 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "ESFDescriptors.h"

#include <glog/logging.h>

using namespace pcl;
using namespace std;

void ESFDescriptor::ExtractDescriptor(const PointCloud<PointXYZI>::Ptr cluster, 
        Eigen::VectorXf& ev_descriptor)
{
    PointCloud<ESFSignature640>::Ptr signature(new PointCloud<ESFSignature640>);

    esf_estimator_.setInputCloud(cluster);
    esf_estimator_.compute(*signature);

    CHECK_EQ(signature->size(), 1u);

    for (size_t i = 0; i < 10; i++)
    {
        int cur_hist_total = 0;
        for (size_t j = 0; j< 64; j ++)
        {
           cur_hist_total += signature->points[0].histogram[i*64 + j]; 
        }
         
        for (size_t j = 0; j < 4; j++) 
        {
            
        }
        
    }
    
    
}
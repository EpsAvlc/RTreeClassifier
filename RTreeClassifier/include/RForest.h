/*
 * Created on Tue Mar 24 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
/*
 * Created on Wed Mar 25 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#ifndef RFOREST_H__
#define RFOREST_H__

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <Eigen/Core>

class RForest
{
public:
    RForest(std::string model_path="");
    void Train(const Eigen::MatrixXf& features, const Eigen::MatrixXf& labels);
    void Test(const Eigen::MatrixXf& features, const Eigen::MatrixXf& label);
    void displayPerformances(unsigned int tp, unsigned int tn,
            unsigned int fp, unsigned int fn);
private:
    cv::Ptr<cv::ml::RTrees> rtrees_;
};

#endif
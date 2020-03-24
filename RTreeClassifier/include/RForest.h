/*
 * Created on Tue Mar 24 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <Eigen/Core>

class RForest
{
public:
    RForest(std::string model_path="");
    void Train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels);
private:
    cv::Ptr<cv::ml::RTrees> rtrees_;
};
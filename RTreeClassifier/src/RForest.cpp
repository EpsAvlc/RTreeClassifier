/*
 * Created on Tue Mar 24 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "RForest.h"
#include <opencv2/ml/ml.hpp>
#include <common.h>

using namespace cv;
using namespace std;
RForest::RForest(string model_path)
{
  if(model_path.empty())
  {
    rtrees_->create();
    rtrees_->setMaxDepth(10);
    rtrees_->setMinSampleCount(10);
    rtrees_->setRegressionAccuracy(0);
    rtrees_->setUseSurrogates(false);
    rtrees_->setMaxCategories(15);
    rtrees_->setPriors(Mat());
    rtrees_->setCalculateVarImportance(true);
    rtrees_->setActiveVarCount(4);
    TermCriteria Tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.01f);
    rtrees_->setTermCriteria(Tc);
  }
  else
  {
    rtrees_ = ml::StatModel::load<ml::RTrees>(model_path);
  }
  
}

void RForest::Train(const Eigen::MatrixXd& features, 
const Eigen::MatrixXd& labels)
{
  const unsigned int n_training_samples = features.rows();
  const unsigned int descriptors_dimension = features.cols();
  LOG(INFO) << "Training RF with " << n_training_samples << " of dimension "
  << descriptors_dimension << ".";

  Mat opencv_features(n_training_samples, descriptors_dimension, CV_32FC1);
  Mat opencv_labels(n_training_samples, 1, CV_32FC1);
  for (unsigned int i = 0u; i < n_training_samples; ++i) {
    for (unsigned int j = 0u; j < descriptors_dimension; ++j) {
      opencv_features.at<float>(i, j) = features(i, j);
    }
    opencv_labels.at<float>(i, 0) = labels(i, 0);
  }

  Ptr<ml::TrainData> tdata;
  tdata->create(opencv_features, ml::SampleTypes::ROW_SAMPLE, opencv_labels);
}
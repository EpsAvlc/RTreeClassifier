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
    rtrees_ = ml::RTrees::create();
    rtrees_->setMaxDepth(20);
    rtrees_->setMinSampleCount(25);
    rtrees_->setRegressionAccuracy(0);
    rtrees_->setUseSurrogates(false);
    rtrees_->setMaxCategories(15);
    rtrees_->setPriors(Mat());
    rtrees_->setCalculateVarImportance(true);
    rtrees_->setUse1SERule(true);
    rtrees_->setActiveVarCount(4);
    TermCriteria Tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.01f);
    rtrees_->setTermCriteria(Tc);
    Mat priors(4, 1, CV_32FC1);
    priors.at<float>(0, 0) = 0.1;
    priors.at<float>(0, 0) = 0.7;
    priors.at<float>(0, 0) = 0.1;
    priors.at<float>(0, 0) = 0.1;
    rtrees_->setPriors(priors);
    // rtrees_->setCVFolds(10);
  }
  else
  {
    rtrees_ = ml::StatModel::load<ml::RTrees>(model_path);
  }
  
}

void RForest::Train(const Eigen::MatrixXf& features, 
const Eigen::MatrixXf& labels)
{
  const unsigned int n_training_samples = features.rows();
  const unsigned int descriptors_dimension = features.cols();
  LOG(INFO) << "Training RF with " << n_training_samples << " of dimension "
  << descriptors_dimension << ".";
  int label_count[6] = {};
  for(int i = 0; i < n_training_samples; i++)
  {
    label_count[(int)labels(i, 0)] ++;
  }
  for(int i = 0; i < 6; i++)
  {
    LOG(INFO) << "Class " << i << " has " << label_count[i] << " samples.";
  }

  Mat opencv_features(n_training_samples, descriptors_dimension, CV_32FC1);
  Mat opencv_labels(n_training_samples, 1, CV_32FC1);
  for (unsigned int i = 0u; i < n_training_samples; ++i) {
    for (unsigned int j = 0u; j < descriptors_dimension; ++j) {
      opencv_features.at<float>(i, j) = features(i, j);
    }
    opencv_labels.at<float>(i, 0) = labels(i, 0);
  }

  Ptr<ml::TrainData> tdata;
  tdata = ml::TrainData::create(opencv_features, ml::SampleTypes::ROW_SAMPLE, opencv_labels);

  rtrees_->train(tdata);
}

void RForest::Test(const Eigen::MatrixXf& features, const Eigen::MatrixXf& labels)
{
  const unsigned int n_samples = features.rows();
  const unsigned int descriptors_dimension = features.cols();
  LOG(INFO)<< "Testing the random forest with " << n_samples
  << " samples of dimension " << descriptors_dimension << ".";

  int label_count[6] = {};
  for(int i = 0; i < n_samples; i++)
  {
    label_count[(int)labels(i, 0)] ++;
  }
  for(int i = 0; i < 6; i++)
  {
    LOG(INFO) << "Class " << i << " has " << label_count[i] << " samples.";
  }

  if (n_samples > 0u) 
  {
    unsigned int tp = 0u, fp = 0u, tn = 0u, fn = 0u;
    for (unsigned int i = 0u; i < n_samples; ++i) {
      Mat opencv_sample(1, descriptors_dimension, CV_32FC1);
      for (unsigned int j = 0u; j < descriptors_dimension; ++j) {
        opencv_sample.at<float>(j) = features(i, j);
      }
      float probability = rtrees_->predict(opencv_sample);
      if (fabs(probability - labels(i, 0)) <= 0.49) {
          ++tp;
        } else {
          ++fp;
        }
    }
    displayPerformances(tp, tn, fp, fn);
  }
}

void RForest::SaveModel(string file_path)
{
  rtrees_->save(file_path);
}

void RForest::displayPerformances(unsigned int tp, unsigned int tn,
                                unsigned int fp, unsigned int fn) {

  LOG(INFO) << "TP: " << tp << ", TN: " << tn <<
      ", FP: " << fp << ", FN: " << fn << ".";

  const double true_positive_rate = double(tp) / double(tp + fn);
  const double true_negative_rate = double(tn) / double(fp + tn);
  const double false_positive_rate = 1.0 - true_negative_rate;

  LOG(INFO) << "Accuracy (ACC): " << double(tp + tn) /
      double(tp + fp + tn + fn);
  LOG(INFO) << "Sensitivity (TPR): " << true_positive_rate;
  LOG(INFO) << "Specificity (TNR): " << true_negative_rate;
  LOG(INFO) << "Precision: " << double(tp) / double(tp + fp);
  LOG(INFO) << "Positive likelyhood ratio: " << true_positive_rate / false_positive_rate;
}

float RForest::Predict(const Eigen::VectorXf& feature)
{
  Mat opencv_sample(1, feature.rows(), CV_32FC1);
  for (unsigned int i = 0u; i < feature.rows(); ++i) 
  {
    opencv_sample.at<float>(i) = feature(i, 0);
  }

  return rtrees_->predict(opencv_sample);
}
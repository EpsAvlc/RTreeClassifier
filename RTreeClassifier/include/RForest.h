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
    /**
     * @brief Construct a new RForest object
     * 
     * @param model_path the saved model path.
     */
    RForest(std::string model_path="");
    /**
     * @brief Train a random forest classifier.
     * 
     * @param features input features. Here it is the cluster's descriptors.
     *      It's a n*7 matrix where n is the numbers of clusters.
     * @param labels input labels of the clusters.
     *      It's a n*1 matrix.
     */
    void Train(const Eigen::MatrixXf& features, const Eigen::MatrixXf& labels);
    /**
     * @brief Test the classifier.
     * 
     * @param features same as train. 
     * @param label same as labels.
     */
    void Test(const Eigen::MatrixXf& features, const Eigen::MatrixXf& label);
    /**
     * @brief Save the model.
     * 
     * @param model_path the path that the model will be saved.
     */
    void SaveModel(const std::string model_path);
    /**
     * @brief Predict a descriptor's class.
     * 
     * @param feature input descriptor of a cluster.
     * @return float output probability.
     */
    float Predict(const Eigen::VectorXf& feature);
private:
    cv::Ptr<cv::ml::RTrees> rtrees_;
    /**
     * @brief display the performance of the classifier. 
     * 
     * @param tp true positive number.
     * @param tn true negative number.
     * @param fp false positive number.
     * @param fn false negative number.
     */
    void displayPerformances(unsigned int tp, unsigned int tn,
        unsigned int fp, unsigned int fn);
};

#endif
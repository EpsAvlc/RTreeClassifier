/*
 * Created on Fri Mar 20 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "EVDescriptor.h"

#include <pcl/common/centroid.h>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace pcl;

void EVDescriptor::ExtractDescriptor(const PointCloud<PointXYZI>::Ptr cluster, 
        Eigen::VectorXf& ev_descriptor)
{
    Eigen::Matrix3f covariance_matrix;
    computeCovarianceMatrix(*cluster, covariance_matrix);

    constexpr bool compute_eigenvectors = false;
    Eigen::EigenSolver<Eigen::Matrix3f> eigenvalues_solver(covariance_matrix, compute_eigenvectors);
    std::vector<float> eigenvalues(3, 0.0);
    eigenvalues.at(0) = eigenvalues_solver.eigenvalues()[0].real();
    eigenvalues.at(1) = eigenvalues_solver.eigenvalues()[1].real();
    eigenvalues.at(2) = eigenvalues_solver.eigenvalues()[2].real();
    if (eigenvalues_solver.eigenvalues()[0].imag() != 0.0 ||
        eigenvalues_solver.eigenvalues()[1].imag() != 0.0 ||
        eigenvalues_solver.eigenvalues()[2].imag() != 0.0 )
    {
    cerr << __FUNCTION__ << ": Eigenvalues should not have non-zero imaginary component." << endl;
    }

    // Sort eigenvalues from smallest to largest.
	sort(eigenvalues.begin(),eigenvalues.end(), 
	[](const float& lhs, const float& rhs)
	{ 
		return lhs < rhs;
	});

    double sum_eigenvalues = eigenvalues.at(0) + eigenvalues.at(1) 
        + eigenvalues.at(2);
    double e1 = eigenvalues.at(0) / sum_eigenvalues;
    double e2 = eigenvalues.at(1) / sum_eigenvalues;
    double e3 = eigenvalues.at(2) / sum_eigenvalues;
    if(e1 == e2 || e2 == e3 || e1 == e3)
    cerr << __FUNCTION__ << ": Eigenvalues should not be equal." << endl;

    // Store inside features.
    const double sum_of_eigenvalues = e1 + e2 + e3;
    constexpr double kOneThird = 1.0/3.0;
    assert(e1 != 0.0);
    assert(sum_of_eigenvalues != 0.0);

    const double kNormalizationPercentile = 1.0;

    const double kLinearityMax = 28890.9 * kNormalizationPercentile;
    const double kPlanarityMax = 95919.2 * kNormalizationPercentile;
    const double kScatteringMax = 124811 * kNormalizationPercentile;
    const double kOmnivarianceMax = 0.278636 * kNormalizationPercentile;
    const double kAnisotropyMax = 124810 * kNormalizationPercentile;
    const double kEigenEntropyMax = 0.956129 * kNormalizationPercentile;
    const double kChangeOfCurvatureMax = 0.99702 * kNormalizationPercentile;

    const double kNPointsMax = 13200 * kNormalizationPercentile;

    ev_descriptor = Eigen::VectorXf::Zero(7);
    // linearity
    ev_descriptor(0) = (e1 - e2) / e1 / kLinearityMax;
    // planarity
    ev_descriptor(1) = (e2 - e3) / e1 / kPlanarityMax;
    // scattering
    ev_descriptor(2) = e3 / e1 / kScatteringMax;
    // omnivariance
    ev_descriptor(3) = pow(e1 * e2 * e3, kOneThird) / kOmnivarianceMax;
    // anisotropy
    ev_descriptor(4) = (e1 - e3) / e1 / kAnisotropyMax;
    // EigenEntropy
    ev_descriptor(5) = (e1 * std::log(e1)) + (e2 * std::log(e2)) + 
        (e3 * std::log(e3)) / kEigenEntropyMax;
    // change of curvature
    ev_descriptor(6) = e3 / sum_of_eigenvalues / kChangeOfCurvatureMax;
}

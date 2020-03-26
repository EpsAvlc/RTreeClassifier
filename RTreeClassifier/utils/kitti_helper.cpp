/*
 * Created on Mon Mar 16 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "kitti_helper.h"

#include <fstream> 
#include <string>

#include <Eigen/Dense>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>

using namespace std;
using namespace pcl;

void KITTIHelper::ReadPointCloud(const std::string& infile, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_ptr)
{
	cloud_ptr.reset(new PointCloud<PointXYZI>);
    fstream input(infile.c_str(), ios::in | ios::binary);
	if(!input.good()){
		cerr << "Could not read file: " << infile << endl;
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);

	for (int i=0; input.good() && !input.eof(); i++) {
		pcl::PointXYZI point;
		input.read((char *) &point.x, 3*sizeof(float));
		input.read((char *) &point.intensity, sizeof(float));
		cloud_ptr->push_back(point);
	}
	input.close();
}

/*
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
*/

void KITTIHelper::ParseLabel(const std::string& infile, std::vector<BBox>& bboxes)
{
	bboxes.clear();
	fstream input(infile.c_str(), ios::in);
	if(!input)
	{
		cerr << __FUNCTION__ << ": invalid lable path" << endl;
		input.close();
	}
	while(!input.eof())
	{
		string bbox_str;
		getline(input, bbox_str);
		if(bbox_str.empty())
			continue;
		vector<string> bbox_strs;
		split(bbox_str, ' ', bbox_strs);
		if(bbox_strs[0].compare("DontCare") == 0)
		{
			continue;
		}
		
		BBox bbox;		
		if(bbox_strs[0].compare("Car") == 0)
		{
			bbox.type = 0;
		}
		// else if(bbox_strs[0].compare("Van") == 0)
		// {
		// 	bbox.type = 1;	
		// }
		// else if(bbox_strs[0].compare("Trunk") == 0)
		// {
		// 	bbox.type = 2;	
		// }
		else if(bbox_strs[0].compare("Pedestrian") == 0)
		{
			bbox.type = 1;
			// bbox.type = 3;	
		}
		else if(bbox_strs[0].compare("Cyclist") == 0)
		{
			bbox.type = 2;	
		}
		else
		{
			bbox.type = 3;
		}

		// If the object is occluded, skip.
		if(atoi(bbox_strs[2].c_str()) > 2)
		{
			continue;
		}

		bbox.height = atof(bbox_strs[8].c_str());
		bbox.width = atof(bbox_strs[9].c_str());
		bbox.length = atof(bbox_strs[10].c_str());

		bbox.x = atof(bbox_strs[11].c_str());
		bbox.y = atof(bbox_strs[12].c_str());
		bbox.z = atof(bbox_strs[13].c_str());
		bbox.rot = atof(bbox_strs[14].c_str());

		// the position in the label is w.r.t the camera frame.
		// Transform it into LiDAR frame

		bboxes.push_back(bbox);
	}
	input.close();
}

void KITTIHelper::split(string& s, char delim, vector<string>& strs)
{
	size_t pos = s.find(delim);
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while( pos != std::string::npos ) {
        strs.push_back( s.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = s.find(delim, initialPos );
    }

    // Add the last one
    strs.push_back( s.substr(initialPos, min( pos, s.size() ) - initialPos + 1 ) );
}

void KITTIHelper::ParseCalib(const std::string& infile, Eigen::MatrixXf& velo_to_cam, Eigen::Matrix3f& R0_rect)
{
	fstream input(infile, ios::in);
	
	string s;
	for(int i = 0; i < 5; i++)
	{
		getline(input, s);
	}
	vector<string> strs;
	split(s, ' ', strs);
	R0_rect = Eigen::Matrix3f::Zero();
	for(int i = 0; i < 9; i++)
	{
		R0_rect(i) = atof(strs[i+1].c_str());
	}
	R0_rect.transposeInPlace();

	getline(input, s);
	split(s, ' ', strs);
	velo_to_cam = Eigen::MatrixXf::Zero(4, 3);
	for(int i = 0; i < 12; i++)
	{
		velo_to_cam(i) = atof(strs[i+1].c_str());
	}
	velo_to_cam.transposeInPlace();
	// cout <<velo_to_cam << endl;
}

void KITTIHelper::ExtractClusters(const string& cloudfile, 
        const string& labelfile, const string& calibfile,
        vector<PointCloud<PointXYZI>::Ptr>& clusters, vector<uint>& labels,
		int size_thres)
{
	PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    ReadPointCloud(cloudfile, cloud); 

	PointCloud<PointXYZI>::Ptr cloud_no_ground(new PointCloud<PointXYZI>);
	RemoveGround(cloud, cloud_no_ground);

    vector<KITTIHelper::BBox> bboxes;
    ParseLabel(labelfile, bboxes);
    
    Eigen::MatrixXf velo2cam;
	Eigen::Matrix3f R0_rect;
    ParseCalib(calibfile, velo2cam, R0_rect);

	Eigen::MatrixXf cam2velo = Eigen::MatrixXf::Zero(3,4);
	cam2velo.block(0, 0, 3, 3) = velo2cam.block(0, 0, 3, 3).transpose();
	cam2velo.block(0, 3, 3, 1) = -velo2cam.block(0, 0, 3, 3).transpose() * 
		velo2cam.block(0, 3, 3, 1);

	// #pragma omp parallel for
	for(int i = 0; i < bboxes.size(); i++)
	{
		float l = bboxes[i].length;
		float w = bboxes[i].width;
		float h = bboxes[i].height;
		float rot = bboxes[i].rot;
		Eigen::MatrixXf pts_rect = Eigen::MatrixXf::Zero(3, 8);

		pts_rect << l * 0.5, l * 0.5, -l * 0.5, -l/2., l/2., l/2., -l/2., -l/2.,
			0, 0, 0, 0, -h, -h, -h, -h,
			w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2;
		// cout << "pts_rect: " << endl << pts_rect << endl; 
		Eigen::Matrix3f rot_rect;
		rot_rect << cos(rot), 0, sin(rot),
				0, 1, 0,
				-sin(rot), 0, cos(rot);
		pts_rect = rot_rect * pts_rect;
		for(int j = 0; j < 8; j++)
		{
			pts_rect(0, j) += bboxes[i].x;
			pts_rect(1, j) += bboxes[i].y;
			pts_rect(2, j) += bboxes[i].z;
		}

		Eigen::MatrixXf pts_ref = R0_rect.inverse() * pts_rect;
		Eigen::MatrixXf pts_l_homo = Eigen::MatrixXf::Ones(4, 8); 

		pts_l_homo.block(0, 0, 3, 8) = pts_ref;
		Eigen::MatrixXf pts_l;
		pts_l = cam2velo * pts_l_homo;

		float x_min = 1000, x_max = -1000, y_min = 1000, 
			y_max = -1000, z_min = 1000, z_max = -1000;
		for(int j = 0; j < 8; j++)
		{
			if(pts_l(0, j) < x_min)
				x_min = pts_l(0, j);
			if(pts_l(1, j) < y_min)
				y_min = pts_l(1, j);
			if(pts_l(2, j) < z_min)
				z_min = pts_l(2, j);
			
			if(pts_l(0, j) > x_max)
				x_max = pts_l(0, j);
			if(pts_l(1, j) > y_max)
				y_max = pts_l(1, j);
			if(pts_l(2, j) > z_max)
				z_max = pts_l(2, j);
		}

		pcl::CropBox<PointXYZI> cb;
		PointCloud<PointXYZI>::Ptr cluster(new PointCloud<PointXYZI>);
		cb.setInputCloud(cloud_no_ground);
		cb.setMin(Eigen::Vector4f(x_min, y_min, z_min, 0));
		cb.setMax(Eigen::Vector4f(x_max, y_max, z_max, 0));
		cb.filter(*cluster);

		if(cluster->size() < size_thres)
			continue;

		clusters.push_back(cluster);
		labels.push_back(bboxes[i].type);
	}
}

void KITTIHelper::RemoveGround(pcl::PointCloud<pcl::PointXYZI>::Ptr ori_cloud, 
	pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud)
{
	sort(ori_cloud->points.begin(), ori_cloud->points.end(), 
	[](const pcl::PointXYZI& lhs, const pcl::PointXYZI& rhs)
	{ 
		return lhs.z < rhs.z;
	});

    double LPR_height = 0; // Lowest Point Represent.
    /* get mean point of LPR */
	int n_lpr = ori_cloud->size() / 5;
    for(int i = 0; i < n_lpr; i ++)
    {
        LPR_height += ori_cloud->points[i].z;
    }
    LPR_height /= n_lpr;

	float k_thres_seeds = 0.5;
	PointCloud<PointXYZI>::Ptr seeds(new PointCloud<PointXYZI>);
    for(int i = 0; i < ori_cloud->points.size(); i++)
    {
        if(isnan(ori_cloud->points[i].x) || isnan(ori_cloud->points[i].y) || isnan(ori_cloud->points[i].z))
        {
            continue;
        }
        if(ori_cloud->points[i].z < LPR_height + k_thres_seeds)
        {
            seeds->push_back(ori_cloud->points[i]);
        }
    }

	// estimatePlane();
	Eigen::Matrix3f cov;
    Eigen::Vector4f mean;
    computeMeanAndCovarianceMatrix(*seeds, cov, mean);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    Eigen::Vector3f norm_vector = svd.matrixU().col(2);
    auto mean_point = mean.head<3>();
    double d = norm_vector.transpose() * mean_point;

	Eigen::Vector3f pt_vec;
	const int k_thres_dist = 0.1;
	for(auto pt:ori_cloud->points) 
	{
		if(isnan(pt.x) || isnan(pt.y) || isnan(pt.z))
			continue;
		pt_vec.x() = pt.x;
		pt_vec.y() = pt.y;
		pt_vec.z() = pt.z;
		if(norm_vector.transpose() * pt_vec - d > k_thres_dist)
			filtered_cloud->points.push_back(pt);
	}
	filtered_cloud->width = filtered_cloud->size();
}
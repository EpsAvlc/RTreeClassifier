/*
 * Created on Mon Mar 16 2020
 *
 * Copyright (c) 2020 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

class KITTIHelper
{
public:
    struct BBox
    {
       int type = -1; 
       float height;
       float width;
       float length;
       float x, y, z;
       float rot; 
    };
    
    /**
     * @brief Read point cloud from kitti .bin file.
     * 
     * @param infile input .bin file
     * @param cloud_ptr output point of the point cloud 
     */
    void ReadPointCloud(const std::string& infile, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_ptr);

    /**
     * @brief Read labels from kitti label file.
     * 
     * @param infile input label file. txt format.
     * @param bboxes output 3D bounding boxes
     */

    void ParseLabel(const std::string& infile, std::vector<BBox>& bboxes);

    /**
     * @brief Read calibration result from kitti calib file.
     * 
     * @param infile input calib file.
     * @param velo_to_cam output transform matrix from velodyne to camera.
     * @param R0_rect output rotation matrix to recity the camera image.
     */
    void ParseCalib(const std::string& infile, Eigen::MatrixXf& velo_to_cam, Eigen::Matrix3f& R0_rect);
    
    /**
     * @brief Combines the above methods to extract point cloud clusters.
     * 
     * @param cloudfile input kitti cloud file.
     * @param labelfile input kitti label file.
     * @param calibfile input kitti calib file.
     * @param clusters output point cloud clusters.
     * @param labels output labels of clusters.
     * @param size_thres threshold to filter clusters. Clusters whose points are
     * less than the threshold will be ignored.
     */
    void ExtractClusters(const std::string& cloudfile, 
        const std::string& labelfile, const std::string& calibfile,
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters,
        std::vector<uint>& labels, int size_thres);

    /**
     * @brief Use the corvariance matrix to estimate and remove ground points.
     * 
     * @param ori_cloud input cloud.
     * @param filtered_cloud out put cloud without ground points.
     */
    void RemoveGround(pcl::PointCloud<pcl::PointXYZI>::Ptr ori_cloud, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud);

private:
    /**
     * @brief split strings into substrings by the delimotor.
     * 
     * @param s input string
     * @param delim delimetor char.
     * @param strs output vectors of substrings.
     */
    void split(std::string& s, char delim, std::vector<std::string>& strs);

};
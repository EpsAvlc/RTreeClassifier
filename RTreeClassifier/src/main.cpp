#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_box.h>
#include <thread>
#include <chrono>

#include "kitti_helper.h"


using namespace std;
using namespace pcl;
int main(int argc, char** argv)
{
	visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();




    KITTIHelper kh;
    string cloudfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/point_cloud/000000.bin";
    
    
    string labelfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/label/000000.txt";
    
    
    string calibfile = "/home/cm/Workspaces/RtreeClassifier/RTreeClassifier/small_training_set/calib/000000.txt";


	// PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    // kh.ReadPointCloud(cloudfile, cloud); 

    // vector<KITTIHelper::BBox> bboxes;
    // kh.ParseLabel(labelfile, bboxes);
    
    // Eigen::MatrixXf velo2cam;
	// Eigen::Matrix3f R0_rect;
    // kh.ParseCalib(calibfile, velo2cam, R0_rect);

	// Eigen::MatrixXf cam2velo = Eigen::MatrixXf::Zero(3,4);
	// cam2velo.block(0, 0, 3, 3) = velo2cam.block(0, 0, 3, 3).transpose();
	// cam2velo.block(0, 3, 3, 1) = -velo2cam.block(0, 0, 3, 3).transpose() * 
	// 	velo2cam.block(0, 3, 3, 1);

// [[ 0.00692796 -0.00116298  0.9999753   0.33219371]
//  [-0.9999722   0.00274984  0.00693114 -0.02210627]
//  [-0.00275783 -0.9999955  -0.0011439  -0.06171977]]

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> single_color(cloud, 255, 0, 0);
    // viewer->addPointCloud<pcl::PointXYZI> (cloud, single_color, "cloud_name");

	vector<PointCloud<PointXYZI>::Ptr> clusters;
	kh.ExtractClusters(cloudfile, labelfile, calibfile, clusters);

	for(int i = 0; i < clusters.size(); i++)
	{
		cout << clusters[i]->size() << endl;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> 
			cluster_color(clusters[i], 255, 0, 0);
    	viewer->addPointCloud<pcl::PointXYZI> (clusters[i], cluster_color, "cluster_" + to_string(i));
	}

// 	for(int i = 0; i < bboxes.size(); i++)
// 	{
// 		// // location in camera frame
// 		// Eigen::Vector3f loc_c(bboxes[i].x, bboxes[i].y, bboxes[i].z);
// 		// // location in LiDAR frame
// 		// Eigen::Vector3f loc_ref = R0_rect.inverse() * loc_c;
// 		// // cout << R0_rect.inverse() << endl;
// 		// Eigen::Vector4f loc_ref_homo(loc_ref.x(), loc_ref.y(), loc_ref.z(), 1);
// 		// Eigen::Vector3f loc_l = cam2velo * loc_ref_homo;
		
// 		// bboxes[i].x = loc_l(0); 
// 		// bboxes[i].y = loc_l(1);
// 		// bboxes[i].z = loc_l(2);

// 		// Eigen::Vector3f min_loc_l;
// 		// min_loc_l.x() = loc_l.x();
// 		// min_loc_l.y() = loc_l.y();
// 		// min_loc_l.z() = loc_l.z() + bboxes[i].height / 2;

// 		// Eigen::Matrix3f rot_c = Eigen::Matrix3f::Identity();
// 		// rot_c(0, 0) = cos(bboxes[i].rot);
// 		// rot_c(0, 2) = sin(bboxes[i].rot);
// 		// rot_c(2, 0) = -sin(bboxes[i].rot);
// 		// rot_c(2, 2) = cos(bboxes[i].rot);

// 		// Eigen::Matrix3f rot_l = cam2velo.block(0, 0, 3, 3) * R0_rect.inverse() * rot_c;
// 		// Eigen::Quaternionf r_q(rot_l);
// 		// //TODO: ｃｕbe的旋转有问题


// 		// pcl::CropBox<PointXYZI> cb;
// 		// PointCloud<PointXYZI>::Ptr cluster(new PointCloud<PointXYZI>);
// 		// cb.setInputCloud(cloud);
// 		// cb.setMin(Eigen::Vector4f(-bboxes[i].length / 2, -bboxes[i].width / 2,
// 		// -bboxes[i].height, 0));
// 		// cb.setMax(Eigen::Vector4f(bboxes[i].length, bboxes[i].width, 
// 		// bboxes[i].height, 0));

// 		// Eigen::Vector3f box_t;
// 		// box_t.x() = loc_l.x();
// 		// box_t.y() = loc_l.y();
// 		// box_t.z() = loc_l.z();

// 		// // Eigen::AngleAxisf rot_a;
// 		// // rot_a.fromRotationMatrix(rotation);
// 		// // cb.setRotation(rot_a);
// 		// // cb.setTranslation(box_t);


// 		// Eigen::Affine3f box_T = Eigen::Affine3f::Identity();
// 		// cout << box_T.matrix() << endl;
// 		// box_T.rotate(rot_l);
// 		// box_T.translate(-box_t);
// 		// cb.setTransform(box_T);
// 		// cb.filter(*cluster);
// 		// cout << "cluster size: " << cluster->size() << endl;

// 		// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> 
// 		// 	cluster_color(cluster, 255, 0, 0);
//     	// viewer->addPointCloud<pcl::PointXYZI> (cluster, cluster_color, "cluster_" + to_string(i));

// 		float l = bboxes[i].length;
// 		float w = bboxes[i].width;
// 		float h = bboxes[i].height;
// 		float rot = bboxes[i].rot;
// 		Eigen::MatrixXf pts_rect = Eigen::MatrixXf::Zero(3, 8);

// 		pts_rect << l * 0.5, l * 0.5, -l * 0.5, -l/2., l/2., l/2., -l/2., -l/2.,
// 			0, 0, 0, 0, -h, -h, -h, -h,
// 			w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2;
// 		// cout << "pts_rect: " << endl << pts_rect << endl; 
// 		Eigen::Matrix3f rot_rect;
// 		rot_rect << cos(rot), 0, sin(rot),
// 				0, 1, 0,
// 				-sin(rot), 0, cos(rot);
// 		pts_rect = rot_rect * pts_rect;
// 		for(int j = 0; j < 8; j++)
// 		{
// 			pts_rect(0, j) += bboxes[i].x;
// 			pts_rect(1, j) += bboxes[i].y;
// 			pts_rect(2, j) += bboxes[i].z;
// 		}
// 		cout << "pts_rect: " << endl << pts_rect << endl; 
// 		// cout << "r0_rect: " << endl << R0_rect << endl; 
// 		// cout << "r0_rect_inverse: " << endl << R0_rect.inverse() << endl; 
// 		Eigen::MatrixXf pts_ref = R0_rect.inverse() * pts_rect;
// 		Eigen::MatrixXf pts_l_homo = Eigen::MatrixXf::Ones(4, 8);
// 		cout << "pts_ref: " << endl << pts_ref << endl; 

// 		pts_l_homo.block(0, 0, 3, 8) = pts_ref;
// 		cout << "pts_l_homo" << endl << pts_l_homo << endl;
// 		Eigen::MatrixXf pts_l;
// 		pts_l = cam2velo * pts_l_homo;
// 		cout << pts_l << endl;

// // [[ 2.32567262  2.32422402 -2.32567262 -2.32422402  2.34061544  2.33916684
// //   -2.3107298  -2.3092812 ]
// //  [ 0.02389142  0.02318624 -0.02389142 -0.02318624 -1.45604274 -1.45674792
// //   -1.50382559 -1.50312041]
// //  [ 0.08352386 -0.08748222 -0.08352386  0.08748222  0.08412148 -0.0868846
// //   -0.08292623  0.08807984]]


// 		// PointCloud<PointXYZI>::Ptr hull_cloud(new PointCloud<PointXYZI>);
// 		// ConvexHull<PointXYZI> cHull;
// 		// cHull.setInputCloud(hull_cloud);
// 		// vector<Vertices> hull_polygons;
// 		// PointCloud<PointXYZI>::Ptr hull_points(new PointCloud<PointXYZI>);
// 		// cHull.reconstruct(*hull_points, hull_polygons);
// 		// CropHull<PointXYZI> cropHullFilter;

// 		// cropHullFilter.setHullIndices(hull_polygons);
// 		// cropHullFilter.setHullCloud(hull_points);
//   		// cropHullFilter.setCropOutside(true); // this will remove points inside the hull

// 		// cropHullFilter.setInputCloud(cloud);
// 		// PointCloud<PointXYZI>::Ptr cluster(new PointCloud<PointXYZI>);
// 		// cropHullFilter.filter(*cluster);

// 		float x_min = 1000, x_max = -1000, y_min = 1000, 
// 			y_max = -1000, z_min = 1000, z_max = -1000;
// 		for(int j = 0; j < 8; j++)
// 		{
// 			if(pts_l(0, j) < x_min)
// 				x_min = pts_l(0, j);
// 			if(pts_l(1, j) < y_min)
// 				y_min = pts_l(1, j);
// 			if(pts_l(2, j) < z_min)
// 				z_min = pts_l(2, j);
			
// 			if(pts_l(0, j) > x_max)
// 				x_max = pts_l(0, j);
// 			if(pts_l(1, j) > y_max)
// 				y_max = pts_l(1, j);
// 			if(pts_l(2, j) > z_max)
// 				z_max = pts_l(2, j);
// 		}

// 		pcl::CropBox<PointXYZI> cb;
// 		PointCloud<PointXYZI>::Ptr cluster(new PointCloud<PointXYZI>);
// 		cb.setInputCloud(cloud);
// 		cb.setMin(Eigen::Vector4f(x_min, y_min, z_min, 0));
// 		cb.setMax(Eigen::Vector4f(x_max, y_max, z_max, 0));
// 		cb.filter(*cluster);

// 		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> 
// 			cluster_color(cluster, 0, 255, 0);
//     	viewer->addPointCloud<pcl::PointXYZI> (cluster, cluster_color, "cluster_" + to_string(i));
// 		cout << cluster->size() << endl;



// 		Eigen::Vector3f loc_c(bboxes[i].x, bboxes[i].y, bboxes[i].z);
// 		// location in LiDAR frame
// 		Eigen::Vector3f loc_ref = R0_rect.inverse() * loc_c;
// 		// cout << R0_rect.inverse() << endl;
// 		Eigen::Vector4f loc_ref_homo(loc_ref.x(), loc_ref.y(), loc_ref.z(), 1);
// 		Eigen::Vector3f loc_l = cam2velo * loc_ref_homo;
		
// 		bboxes[i].x = loc_l(0); 
// 		bboxes[i].y = loc_l(1);
// 		bboxes[i].z = loc_l(2);

// 		Eigen::Vector3f min_loc_l;
// 		min_loc_l.x() = loc_l.x();
// 		min_loc_l.y() = loc_l.y();
// 		min_loc_l.z() = loc_l.z() + bboxes[i].height / 2;



// 		Eigen::Matrix3f rot_c = Eigen::Matrix3f::Identity();
// 		rot_c(0, 0) = cos(bboxes[i].rot);
// 		rot_c(0, 2) = sin(bboxes[i].rot);
// 		rot_c(2, 0) = -sin(bboxes[i].rot);
// 		rot_c(2, 2) = cos(bboxes[i].rot);

// 		Eigen::Matrix3f rot_l = cam2velo.block(0, 0, 3, 3) * R0_rect.inverse() * rot_c;
// 		Eigen::Quaternionf r_q(rot_l);

// 		string cube_name = "cube_" + to_string(i);
// 		viewer->addCube(min_loc_l, r_q, bboxes[i].length,bboxes[i].width, bboxes[i].height, cube_name); 
// 	}



    while(!viewer->wasStopped())
    {
        viewer->spinOnce (100);
    	std::this_thread::sleep_for(chrono::milliseconds(30));
    }
    return 0;
}
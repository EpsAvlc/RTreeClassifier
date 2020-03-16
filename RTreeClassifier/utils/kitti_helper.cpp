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

using namespace std;

void KITTIHelper::ReadPointCloud(const std::string& infile, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_ptr)
{
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

void KITTIHelper::ParseLabel(const std::string& infile, vector<BBox>& bboxes)
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
		else if(bbox_strs[0].compare("Van") == 0)
		{
			bbox.type = 1;	
		}
		else if(bbox_strs[0].compare("Trunk") == 0)
		{
			bbox.type = 2;	
		}
		else if(bbox_strs[0].compare("Pedestrian") == 0)
		{
			bbox.type = 3;	
		}
		else if(bbox_strs[0].compare("Cyclist") == 0)
		{
			bbox.type = 4;	
		}
		else
		{
			continue;
		}

		// If the object is occluded, skip.
		if(atoi(bbox_strs[2].c_str()) != 0)
		{
			continue;
		}

		bbox.height = atof(bbox_strs[8].c_str());

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
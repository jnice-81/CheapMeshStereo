#include <opencv2/opencv.hpp>

/*
Storage class for voxels used in Scene.

position: The position
normal: The orientation of the point
numhits: The number of times this voxel was hit (inserted into). 
*/
class ScenePoint {
public:
	ScenePoint(cv::Vec3f position, cv::Vec3f normal, short numhits)
	{
		this->position = position;
		this->normal = normal;
		this->numhits = numhits;
	}

	ScenePoint() {

	}

	cv::Vec3f position;
	cv::Vec3f normal;
	short numhits;
};

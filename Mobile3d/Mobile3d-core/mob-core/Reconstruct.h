#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <list>

#include "View.h"

class Reconstruct {
public:
	static void OpenGL2OpenCVView(View& v) {
		cv::Mat_<double> ogl_to_cv = cv::Mat_<double>(4, 4);
		ogl_to_cv <<
			1, 1, 1, 1,
			-1, -1, -1, -1,
			-1, -1, -1, -1,
			0, 0, 0, 1;
		cv::Size imgsize = v.image.size();

		v.extrinsics = v.extrinsics.mul(ogl_to_cv);
		cv::Mat_<double> cvintrinsics = cv::Mat_<double>(3, 3);
		cvintrinsics <<
			v.intrinsics.at<double>(0, 0) * 0.5 * imgsize.width, 0, v.intrinsics.at<double>(0, 2) + imgsize.width * 0.5,
			0, v.intrinsics.at<double>(1, 1) * 0.5 * imgsize.height, v.intrinsics.at<double>(1, 2) + imgsize.height * 0.5,
			0, 0, 1;
		v.intrinsics = cvintrinsics;
	}

	void add_image(View new_view) {
		sliding_window.push_back(new_view);

		if (sliding_window.size() >= 2) {
			update3d();
			sliding_window.pop_front();
		}
	}

private:
	void update3d() {
		View v1 = sliding_window.front();
		View v2 = sliding_window.back();

		cv::Rect roiR = cv::Rect(0, 0, 3, 3);
		cv::Rect roiT = cv::Rect(3, 0, 1, 3);
		cv::Mat R = v2.extrinsics(roiR) * v1.extrinsics(roiR).t();
		cv::Mat T = v2.extrinsics(roiT) - R * v1.extrinsics(roiT);

		cv::Size imgsize = v1.image.size();
		cv::Mat rR1, rR2, rP1, rP2, Q;

		cv::stereoRectify(v1.intrinsics, cv::Mat(), v2.intrinsics, cv::Mat(), imgsize,
			R, T, rR1, rR2, rP1, rP2, Q, cv::CALIB_ZERO_DISPARITY);

		cv::Mat map1x, map1y, map2x, map2y;
		cv::initUndistortRectifyMap(v1.intrinsics, cv::Mat(), rR1, rP1, imgsize, CV_16SC2, map1x, map1y);
		cv::initUndistortRectifyMap(v2.intrinsics, cv::Mat(), rR2, rP2, imgsize, CV_16SC2, map2x, map2y);

		cv::Mat rectified_image1, rectified_image2;
		cv::remap(v1.image, rectified_image1, map1x, map1y, cv::INTER_LINEAR);
		cv::remap(v2.image, rectified_image2, map2x, map2y, cv::INTER_LINEAR);

		cv::imshow("v1", rectified_image1);
		cv::imshow("v2", rectified_image2);


		
		cv::waitKey(0);
	}

	std::list<View> sliding_window;
};
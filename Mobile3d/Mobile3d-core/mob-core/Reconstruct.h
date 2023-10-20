#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <list>
#include <chrono>

#pragma warning(push, 0)
#include <PreProcessor.h>
#include <Reconstructors.h>
#pragma warning(pop)

#include "helpers.h"
//#include "PyramidalBM.h"
#include "View.h"
#include "Scene.h"


class Reconstruct {
public:
	Reconstruct(Scene& s) : scene(s) {}

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

	bool shouldAddImage(View newView, float minNorm) {
		const cv::Rect roiR = cv::Rect(0, 0, 3, 3);
		const cv::Rect roiT = cv::Rect(3, 0, 1, 3);

		if (sliding_window.size() == 0) {
			return true;
		}

		View oldView = sliding_window.back();
		cv::Mat R = newView.extrinsics(roiR) * oldView.extrinsics(roiR).t();
		cv::Mat T = newView.extrinsics(roiT) - R * oldView.extrinsics(roiT);

		return cv::norm(T) >= minNorm;
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

		MsClock csc;

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

		csc.printAndReset("Rectify");
		/*
		cv::imshow("v1", rectified_image1);
		cv::imshow("v2", rectified_image2);
		cv::waitKey(0);
		*/

		int ndisp = 30 * 16;
		int mindisp = -(ndisp / 2);
		cv::Ptr<cv::StereoBM> blocksearcher = cv::StereoBM::create(ndisp, 21);
		blocksearcher->setMinDisparity(mindisp);
		
		cv::cvtColor(rectified_image1, rectified_image1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(rectified_image2, rectified_image2, cv::COLOR_BGR2GRAY);
		cv::Mat disparity;
		blocksearcher->compute(rectified_image1, rectified_image2, disparity);
		disparity /= 16;

		csc.printAndReset("Disparity");

		addDisparity(disparity, Q, rR1, v1.extrinsics, mindisp - 1);

		csc.printAndReset("Add3d");
	}

	inline int virtualToRealNormalBufferIdx(int i, int currentWriteLine, int bufferLines) {
		return (currentWriteLine + i) % bufferLines;
	}

	void addDisparity(const cv::Mat &disparity, const cv::Mat &Q, const cv::Mat &Rrectify, const cv::Mat &extrinsics, const int undefined) {
		assert(disparity.type() == CV_16S);

		cv::Mat Rrectify4x4 = cv::Mat::zeros(4, 4, CV_64F);
		Rrectify4x4(cv::Rect(0, 0, 3, 3)) = Rrectify.t();
		Rrectify4x4.at<double>(3, 3) = 1;
		cv::Mat tmp = extrinsics.inv() * Rrectify4x4 * Q;
		cv::Matx44d Pback;
		tmp.convertTo(Pback, CV_64F);

		std::vector<std::vector<cv::Vec3d>> normalBufferPoints;
		std::vector<std::vector<bool>> isNormalBufferDefined;
		const int bufferLines = 3;
		const int minimumAvailable = 2;
		normalBufferPoints.resize(bufferLines);
		isNormalBufferDefined.resize(bufferLines);
		for (int i = 0; i < bufferLines; i++) {
			normalBufferPoints[i].resize(disparity.cols);
			isNormalBufferDefined[i].resize(disparity.cols);
		}
		int currentWriteLine = 0;
		bool isBufferFilled = false;

		for (int y = 0; y < disparity.rows; y++) {
			const short* rptr = disparity.ptr<short>(y);
			for (int x = 0; x < disparity.cols; x++) {
				short disp = rptr[x];
				if (disp != 0 && disp != undefined) {
					cv::Vec4d cam = Pback * cv::Vec4d((double)x, (double)y, (double)disp, 1.0);
					cv::Vec3d pos = cv::Vec3d(cam.val) / cam[3];

					normalBufferPoints[currentWriteLine][x] = pos;
					isNormalBufferDefined[currentWriteLine][x] = true;
					//scene.addPoint(pos);
				}
				else {
					isNormalBufferDefined[currentWriteLine][x] = false;
				}
			}

			currentWriteLine = (currentWriteLine + 1) % bufferLines;
			isBufferFilled = isBufferFilled || currentWriteLine == 0;
			if (isBufferFilled) {

				for (int j = 0; j < disparity.cols - bufferLines; j++) {
					int currentIdxY = virtualToRealNormalBufferIdx(bufferLines / 2, currentWriteLine, bufferLines);
					int currentIdxX = j + bufferLines / 2;
					if (!isNormalBufferDefined[currentIdxY][currentIdxX]) {
						continue;
					}

					cv::Vec3f left = vecZeros<cv::Vec3f>();
					cv::Vec3f bottom = vecZeros<cv::Vec3f>();
					int countDefLeft = 0;
					int countDefBottom = 0;

					for (int k = 0; k < bufferLines; k++) {
						int yidx = virtualToRealNormalBufferIdx(k, currentWriteLine, bufferLines);
						for (int z = 0; z < bufferLines -1; z++) {
							int xidx = z + j;
							int nextX = xidx + 1;

							if (isNormalBufferDefined[yidx][xidx] && isNormalBufferDefined[yidx][nextX]) {
								cv::Vec3d c = normalBufferPoints[yidx][xidx] - normalBufferPoints[yidx][nextX];
								left += c / cv::norm(c);
								countDefLeft += 1;
							}
						}
					}

					for (int k = 0; k < bufferLines-1; k++) {
						int yidx = virtualToRealNormalBufferIdx(k, currentWriteLine, bufferLines);
						int nextY = virtualToRealNormalBufferIdx(k+1, currentWriteLine, bufferLines);
						for (int z = 0; z < bufferLines; z++) {
							int xidx = z + j;

							if (isNormalBufferDefined[yidx][xidx] && isNormalBufferDefined[nextY][xidx]) {
								cv::Vec3d c = normalBufferPoints[yidx][xidx] - normalBufferPoints[nextY][xidx];
								bottom += c / cv::norm(c);
								countDefBottom += 1;
							}
						}
					}

					if (countDefBottom >= minimumAvailable && countDefLeft >= minimumAvailable) {
						left /= countDefLeft;
						bottom /= countDefBottom;

						cv::Vec3f n = - left.cross(bottom);
						n = n / cv::norm(n);
						cv::Vec3f p = normalBufferPoints[currentIdxY][currentIdxX];

						scene.addPoint(p, n);
					}
				}
			}
		}
	}

	std::list<View> sliding_window;
	Scene &scene;
};
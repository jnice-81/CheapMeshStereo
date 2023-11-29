#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <list>
#include <chrono>

#include "helpers.h"
//#include "PyramidalBM.h"
#include "View.h"
#include "Scene.h"
//#include "PoissonSurfaceReconstruct.h"

class Reconstruct {
public:
	Reconstruct() {}

	bool shouldAddImage(const cv::Mat &newExtrinsics, float minNorm) {
		const cv::Rect roiR = cv::Rect(0, 0, 3, 3);
		const cv::Rect roiT = cv::Rect(3, 0, 1, 3);

		if (sliding_window.size() == 0) {
			return true;
		}

		View oldView = sliding_window.back();
		cv::Mat R = newExtrinsics(roiR) * oldView.extrinsics(roiR).t();
		cv::Mat T = newExtrinsics(roiT) - R * oldView.extrinsics(roiT);

		return cv::norm(T) >= minNorm;
	}

	void add_image(View new_view) {
		sliding_window.push_back(new_view);

		if (sliding_window.size() > 2) {
			sliding_window.pop_front();
		}
	}

	void update3d(std::vector<ScenePoint> &out) {
        if (sliding_window.size() < 2) {
            return;
        }

		View v1 = sliding_window.front();
		View v2 = sliding_window.back();

		MsClock csc;

		cv::Rect roiR = cv::Rect(0, 0, 3, 3);
		cv::Rect roiT = cv::Rect(3, 0, 1, 3);
		cv::Mat R = v2.extrinsics(roiR) * v1.extrinsics(roiR).t();
		cv::Mat T = v2.extrinsics(roiT) - R * v1.extrinsics(roiT);

		cv::Size imgsize = v1.image.size();
		cv::Mat rR1, rR2, rP1, rP2, Q;
		cv::Rect validRoiV1, validRoiV2;

		cv::stereoRectify(v1.intrinsics, cv::Mat(), v2.intrinsics, cv::Mat(), imgsize,
			R, T, rR1, rR2, rP1, rP2, Q, cv::CALIB_ZERO_DISPARITY, -1, cv::Size(), &validRoiV1, &validRoiV2);

		if (rP2.at<double>(1, 3) != 0) {
			std::cout << "Aborted update because of vertical shift";
			return; // Indicates vertical rectification which is not supported for now
		}
		double shiftedTo = rP2.at<double>(0, 3) / rP2.at<double>(0, 0);

		cv::Mat map1x, map1y, map2x, map2y;
		cv::initUndistortRectifyMap(v1.intrinsics, cv::Mat(), rR1, rP1, imgsize, CV_16SC2, map1x, map1y);
		cv::initUndistortRectifyMap(v2.intrinsics, cv::Mat(), rR2, rP2, imgsize, CV_16SC2, map2x, map2y);

		cv::Mat rectified_image1, rectified_image2;
		cv::remap(v1.image, rectified_image1, map1x, map1y, cv::INTER_LINEAR);
		cv::remap(v2.image, rectified_image2, map2x, map2y, cv::INTER_LINEAR);

		csc.printAndReset("Rectify");

		/*
		cv::imwrite("/data/data/com.google.ar.core.examples.c.helloar/v1.jpg", rectified_image1);
		cv::imwrite("/data/data/com.google.ar.core.examples.c.helloar/v2.jpg", rectified_image2);
		*/

		int ndisp = 15 * 16;
		int mindisp;
		if (shiftedTo > 0) {
			mindisp = -15 * 16;
		}
		else {
			mindisp = 0;
		}
		
		cv::Ptr<cv::StereoBM> blocksearcher = cv::StereoBM::create(ndisp, 21);
		blocksearcher->setMinDisparity(mindisp);
		blocksearcher->setUniquenessRatio(15);
		// There is a bug with the disparity matching of opencv generating weird structured
		// patterns in the disparity map.
		//LOGI("%d, %d, %d, %d", validRoiV1.x, validRoiV1.y, validRoiV1.width, validRoiV1.height);
		//LOGI("%d, %d, %d, %d", validRoiV2.x, validRoiV2.y, validRoiV2.width, validRoiV2.height);
		//blocksearcher->setROI1(validRoiV1);
		//blocksearcher->setROI2(validRoiV2);
		
		cv::cvtColor(rectified_image1, rectified_image1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(rectified_image2, rectified_image2, cv::COLOR_BGR2GRAY);
		cv::Mat disparity;
		blocksearcher->compute(rectified_image1, rectified_image2, disparity);
		disparity /= 16;

		csc.printAndReset("Disparity");

		/*
		cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX);
		disparity.convertTo(disparity, CV_8U);
		cv::imshow("v1", rectified_image1);
		cv::imshow("v2", rectified_image2);
		cv::imshow("disp", disparity);
		cv::waitKey(0);
		*/

		addDisparity(disparity, Q, rR1, v1.extrinsics, mindisp - 1, out);

		csc.printAndReset("Add3d");
	}

private:
	inline int virtualToRealNormalBufferIdx(int i, int currentWriteLine, int bufferLines) {
		return (currentWriteLine + i) % bufferLines;
	}

	void addDisparity(const cv::Mat &disparity, const cv::Mat &Q, const cv::Mat &Rrectify, const cv::Mat &extrinsics, const int undefined, std::vector<ScenePoint> &out) {
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

						out.emplace_back(p, n, 1.0f);
					}
				}
			}
		}
	}

	std::list<View> sliding_window;
};
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>

#include "helpers.h"
#include "View.h"
#include "Scene.h"


/*
A class that supports finding depth points between two images. Images need to have Intrinsics and Extrinsics defined.
*/
class Reconstruct {
public:
	/*
	Get the disparity that is associated with a particular depth
	f: The focal length
	T: The baseline (shift) between two cameras
	*/
	static inline double getDisparityForDepth(const double depth, const double f, const double T) {
		/*
		depth = f * T / disparity
		*/
		return f * T / depth;
	}

	/*
	Get the depth for a particular disparity
	f: The focal length
	T: The baseline (shift) between two cameras
	*/
	static inline double getDepthForDisparity(const double disparity, const double f, const double T) {
		return f * T / disparity;
	}

	/*
		Returns the minimum disparity to obtain a certain level of precision.
		f: The focal length
		T: The baseline (shift) between two cameras
		precision: The required precision
	*/
	static inline double getMinDisparityForPrecision(const double f, const double T, const double precision) {
		/*
		depth = f * T / disparity
		g >= f * T / d - f * T / (d+1) = (f * T * (d + 1)) / (d * (d+1)) - (f * T * d) / (d * (d+1)) = (f * T) / (d * (d + 1)) =>
		g * d^2 + g * d - f * T >= 0
		*/
		return (-precision + std::sqrt(precision * precision + 4 * precision * f * T)) / (2 * precision);
	}

	/*
		Essentially: Rectifies to views, computes a disparity image using block matching and projects the points back to 3d. 
		Only horizontal planar motion is supported, but in principle vertical planar motion could be integrated relativly easily.
		Method will simply return without adding points if the motion type is not supported.
		v1: The first view
		v2: The second view
		out: A vector to which all detected points are added (in 3d global coordinate system)
		minDepth: The minimum depth to support
		maxDepth: The maximum depth to support
		precision: The minimum precision required for a point to be added. For a given baseline between two images this effectively 
			truncates the maximum distance for which points will be added (because for points farther away the maximum possible error
			is larger than precision
		maxDisp: The maximum allowed disparity. The actual maximum disparity is controlled by minDepth, but maxDisp acts as a hard threshold.
			Must be dividable by 16.
	*/
	static void compute3d(const View &v1, const View &v2, std::vector<ScenePoint> &out,
		double minDepth, double maxDepth, double precision, size_t maxDisp = 16 * 15) {

		cv::Rect roiR = cv::Rect(0, 0, 3, 3);
		cv::Rect roiT = cv::Rect(3, 0, 1, 3);
		cv::Mat R = v2.extrinsics(roiR) * v1.extrinsics(roiR).t();
		cv::Mat T = v2.extrinsics(roiT) - R * v1.extrinsics(roiT);

		cv::Size imgsize = v1.image.size();
		cv::Mat rR1, rR2, rP1, rP2, Q;
		cv::Rect validRoiV1, validRoiV2;

		cv::stereoRectify(v1.intrinsics, cv::Mat(), v2.intrinsics, cv::Mat(), imgsize,
			R, T, rR1, rR2, rP1, rP2, Q, cv::CALIB_ZERO_DISPARITY, -1, cv::Size(), &validRoiV1, &validRoiV2);

		int x = std::max(validRoiV1.x, validRoiV2.x);
		int y = std::max(validRoiV1.y, validRoiV2.y);
		int width = std::min(validRoiV1.x + validRoiV1.width, validRoiV2.x + validRoiV2.width) - x;
		int height = std::min(validRoiV1.y + validRoiV1.height, validRoiV2.y + validRoiV2.height) - y;
		cv::Rect disparityRoi = cv::Rect(x, y, width, height);

		if (rP2.at<double>(1, 3) != 0) {
			std::cout << "Aborted computation because of vertical shift (not supported for now)";
			return;
		}
		double shift = -1 / Q.at<double>(3, 2);
		double f = Q.at<double>(2, 3);

		int maxDispDepth = (int)getDisparityForDepth(minDepth, f, std::abs(shift));
		int minDispPrec = (int)getMinDisparityForPrecision(f, std::abs(shift), precision);
		int minDispDepth = (int)getDisparityForDepth(maxDepth, f, std::abs(shift));
		int minDisp = std::max(minDispPrec, minDispDepth);
		maxDispDepth -= maxDispDepth % 16;
		if (maxDispDepth > maxDisp) {
			maxDispDepth = maxDisp;
		}

		if (minDisp >= maxDispDepth) {
			std::cout << "No valid disparity for the given precision and depth. Aborted";
			return;
		}

		if (width - maxDispDepth <= 50 || height <= 50) {
			std::cout << "Aborted because of no usable rectified area";
			return;
		}

		cv::Mat map1x, map1y, map2x, map2y;
		cv::initUndistortRectifyMap(v1.intrinsics, cv::Mat(), rR1, rP1, imgsize, CV_16SC2, map1x, map1y);
		cv::initUndistortRectifyMap(v2.intrinsics, cv::Mat(), rR2, rP2, imgsize, CV_16SC2, map2x, map2y);

		cv::Mat rectified_image1, rectified_image2;
		cv::remap(v1.image, rectified_image1, map1x, map1y, cv::INTER_LINEAR);
		cv::remap(v2.image, rectified_image2, map2x, map2y, cv::INTER_LINEAR);

		rectified_image1 = rectified_image1(disparityRoi);
		rectified_image2 = rectified_image2(disparityRoi);

		int ndisp = maxDispDepth;
		int mindisp;
		if (shift > 0) {
			mindisp = -ndisp;
		}
		else {
			mindisp = 0;
		}

		cv::Ptr<cv::StereoBM> blocksearcher = cv::StereoBM::create(
				ndisp,
				15);
		blocksearcher->setUniquenessRatio(20);
		blocksearcher->setMinDisparity(mindisp);
		blocksearcher->setDisp12MaxDiff(1);
		blocksearcher->setPreFilterSize(9);
		blocksearcher->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
		blocksearcher->setPreFilterCap(31);
		blocksearcher->setTextureThreshold(10);
		blocksearcher->setSpeckleRange(32);
		blocksearcher->setSpeckleWindowSize(100);
		
		cv::Mat disparity;
		blocksearcher->compute(rectified_image1, rectified_image2, disparity);
		disparity /= 16;

		addDisparity(disparity, Q, rR1, v1.extrinsics, minDisp, mindisp-1, out, disparityRoi.x, disparityRoi.y);
	}

private:
	static inline int virtualToRealNormalBufferIdx(int i, int currentWriteLine, int bufferLines) {
		return (currentWriteLine + i) % bufferLines;
	}

	/*
		Reprojects disparity, estimates normals based on a very simple plane fitting approach, and adds points to out
		disparity: The disparity image
		Q: The disparity to depth matrix
		Rrectify: the rectification matrix associated with the image from which the disparities are generated
		extrinsics: The extrinsics matrix associated with the image from which the disparities are generated
		minDisp: The minimum disparity for which points will be added
		undefined: A number indicating which disparity value corresponds to undefined (no disparity found)
		out: The vector of ScenePoints which is filled by the method
		addX: A value to add to the x component of the pixel coordinate when reprojecting (Usefull if parts of the image were truncated)
		addY: A value to add to the y component of the pixel coordinate when reprojecting (Usefull if parts of the image were truncated)
	*/
	static void addDisparity(const cv::Mat& disparity, const cv::Mat& Q, const cv::Mat& Rrectify, const cv::Mat& extrinsics,
		const int minDisp, const int undefined, std::vector<ScenePoint>& out, int addX, int addY) {
		assert(disparity.type() == CV_16S);

		cv::Mat Rrectify4x4 = cv::Mat::zeros(4, 4, CV_64F);
		Rrectify4x4(cv::Rect(0, 0, 3, 3)) = Rrectify.t();
		Rrectify4x4.at<double>(3, 3) = 1;
		cv::Mat tmp = extrinsics.inv() * Rrectify4x4 * Q;
		cv::Matx44d Pback;
		tmp.convertTo(Pback, CV_64F);

		std::vector<std::vector<cv::Vec3d>> normalBufferPoints;
		std::vector<std::vector<bool>> isNormalBufferDefined;
		const int bufferLines = 7;
		const int minimumAvailable = 30;
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
				if (disp != undefined && std::abs(disp) >= minDisp) {
					cv::Vec4d cam = Pback * cv::Vec4d((double)(x + addX), (double)(y + addY), (double)disp, 1.0);
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

					cv::Vec3d forPoint = normalBufferPoints[currentIdxY][currentIdxX];
					cv::Vec3d left = vecZeros<cv::Vec3d>();
					cv::Vec3d bottom = vecZeros<cv::Vec3d>();
					int countDefLeft = 0;
					int countDefBottom = 0;

					for (int k = 0; k < bufferLines; k++) {
						int yidx = virtualToRealNormalBufferIdx(k, currentWriteLine, bufferLines);
						for (int z = 0; z < bufferLines - 1; z++) {
							int xidx = z + j;
							int nextX = xidx + 1;

							if (isNormalBufferDefined[yidx][xidx] && isNormalBufferDefined[yidx][nextX]) {
								cv::Vec3d c = normalBufferPoints[yidx][xidx] - normalBufferPoints[yidx][nextX];								
								left += c / cv::norm(c);
								countDefLeft += 1;
							}
						}
					}

					for (int k = 0; k < bufferLines - 1; k++) {
						int yidx = virtualToRealNormalBufferIdx(k, currentWriteLine, bufferLines);
						int nextY = virtualToRealNormalBufferIdx(k + 1, currentWriteLine, bufferLines);
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

						cv::Vec3f n = -left.cross(bottom);
						float normn = cv::norm(n);
						n = n / normn;
						cv::Vec3f p = normalBufferPoints[currentIdxY][currentIdxX];

						if (normn > 0) {
							out.emplace_back(p, n, 1.0f);
						}
					}
				}
			}
		}
	}


};
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <list>
#include <chrono>

#include "helpers.h"
#include "View.h"
#include "Scene.h"
//#include "PoissonSurfaceReconstruct.h"

class Reconstruct {
public:
	static inline double getDisparityForDepth(const double depth, const double f, const double T) {
		/*
		depth = f * T / disparity
		*/
		return f * T / depth;
	}

	static inline double getDepthForDisparity(const double disparity, const double f, const double T) {
		return f * T / disparity;
	}

	static inline double getMinDisparityForPrecision(const double f, const double T, const double precision) {
		/*
		depth = f * T / disparity
		g >= f * T / d - f * T / (d+1) = (f * T * (d + 1)) / (d * (d+1)) - (f * T * d) / (d * (d+1)) = (f * T) / (d * (d + 1)) =>
		g * d^2 + g * d - f * T >= 0
		*/
		return (-precision + std::sqrt(precision * precision + 4 * precision * f * T)) / (2 * precision);
	}

	static void compute3d(const View &v1, const View &v2, std::vector<ScenePoint> &out,
		double minDepth, double maxDepth, double precision, size_t maxDisp = 16 * 15) {

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

		int x = std::max(validRoiV1.x, validRoiV2.x);
		int y = std::max(validRoiV1.y, validRoiV2.y);
		int width = std::min(validRoiV1.x + validRoiV1.width, validRoiV2.x + validRoiV2.width) - x;
		int height = std::min(validRoiV1.y + validRoiV1.height, validRoiV2.y + validRoiV2.height) - y;
		cv::Rect disparityRoi = cv::Rect(x, y, width, height);

		if (rP2.at<double>(1, 3) != 0) {
			std::cout << "Aborted update because of vertical shift";
			return; // Indicates vertical rectification which is not supported for now
		}
		double shift = -1 / Q.at<double>(3, 2);
		double f = Q.at<double>(2, 3);

		int maxDispDepth = (int)getDisparityForDepth(minDepth, f, std::abs(shift));
		int minDispPrec = (int)getMinDisparityForPrecision(f, std::abs(shift), precision);
		maxDispDepth -= maxDispDepth % 16;
		if (maxDispDepth > maxDisp) {
			maxDispDepth = maxDisp;
		}
		std::cout << minDispPrec << " " << maxDispDepth << " " << shift << std::endl;

		if (minDispPrec >= maxDispDepth) {
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

		csc.printAndReset("Rectify");

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

		cv::cvtColor(rectified_image1, rectified_image1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(rectified_image2, rectified_image2, cv::COLOR_BGR2GRAY);

#ifdef DEBUG_ANDROID
		cv::imwrite("/data/data/com.google.ar.core.examples.c.helloar/vl" + std::to_string(dbgidx) + ".jpg", rectified_image1(disparityRoi));
		cv::imwrite("/data/data/com.google.ar.core.examples.c.helloar/vr" + std::to_string(dbgidx) + ".jpg", rectified_image2(disparityRoi));
#endif

		
		cv::Mat disparity;
		blocksearcher->compute(rectified_image1, rectified_image2, disparity);
		disparity /= 16;

		csc.printAndReset("Disparity");
		

#ifdef DEBUG_ANDROID
		cv::Mat exportDisp;
		cv::normalize(disparity, exportDisp, 0, 255, cv::NORM_MINMAX);
		disparity.convertTo(exportDisp, CV_8U);
		cv::imwrite("/data/data/com.google.ar.core.examples.c.helloar/vdisp" + std::to_string(dbgidx) + ".jpg", exportDisp);
#endif

		//std::vector<ScenePoint> tmpOut;
		addDisparity(disparity, Q, rR1, v1.extrinsics, minDispPrec, mindisp-1, out, disparityRoi.x, disparityRoi.y);

		/*
		Scene<1, bool> u(0.01, std::vector<int>({ 5 }));
		for (const ScenePoint& g : tmpOut) {
			u.addPoint(g);
		}
		u.export_xyz("tmp.xyz");
		
		csc.printAndReset("Add3d");

		cv::imshow("l", rectified_image1);
		cv::imshow("r", rectified_image2);
		cv::Mat exportDisp;
		disparity.convertTo(exportDisp, CV_32F);
		for (int i = 0; i < exportDisp.rows; i++) {
			for (int j = 0; j < exportDisp.cols; j++) {
				double d;
				if (std::abs(exportDisp.at<float>(i, j)) < minDispPrec || exportDisp.at<float>(i, j) == mindisp - 1) {
					d = 0;
				}
				else {
					d = getDepthForDisparity(std::abs(exportDisp.at<float>(i, j)), f, std::abs(shift));
				}
				exportDisp.at<float>(i, j) = d;
			}
			}
		cv::normalize(exportDisp, exportDisp, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("disp", exportDisp);
		cv::waitKey(0);
		*/

	}

private:
	static inline int virtualToRealNormalBufferIdx(int i, int currentWriteLine, int bufferLines) {
		return (currentWriteLine + i) % bufferLines;
	}

	static void addDisparity(const cv::Mat &disparity, const cv::Mat &Q, const cv::Mat &Rrectify, const cv::Mat &extrinsics, 
		const int minDisp, const int undefined, std::vector<ScenePoint> &out, int addX, int addY) {
		assert(disparity.type() == CV_16S);

		cv::Mat Rrectify4x4 = cv::Mat::zeros(4, 4, CV_64F);
		Rrectify4x4(cv::Rect(0, 0, 3, 3)) = Rrectify.t();
		Rrectify4x4.at<double>(3, 3) = 1;
		cv::Mat tmp = extrinsics.inv() * Rrectify4x4 * Q;
		cv::Vec4d camPosTmp;
		((cv::Mat)extrinsics.inv().col(3)).convertTo(camPosTmp, CV_64F);
		cv::Vec3d camPosTmp2(camPosTmp.val);
		cv::Vec3f camPos = camPosTmp2;
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
					std::vector<cv::Vec3f> points;

					for (int k = 0; k < bufferLines; k++) {
						int yidx = virtualToRealNormalBufferIdx(k, currentWriteLine, bufferLines);
						for (int z = 0; z < bufferLines -1; z++) {
							int xidx = z + j;

							if (isNormalBufferDefined[yidx][xidx])  {
								points.push_back(normalBufferPoints[yidx][xidx]);
							}
						}
					}

					if (points.size() >= minimumAvailable) {

						cv::Vec3d mean = cv::Vec3d::zeros();
						for (const auto& g : points) {
							mean += g;
						}
						mean = mean * (1.0 / points.size());
						cv::Mat A(3, points.size(), CV_32F);
						for (int i = 0; i < points.size(); i++) {
							cv::Vec3f p = points[i] - (cv::Vec3f)mean;
							A.at<float>(0, i) = p[0];
							A.at<float>(1, i) = p[1];
							A.at<float>(2, i) = p[2];
						}

						cv::SVD svd(A);

						cv::Vec3f n = svd.u.col(2);
						n = n / cv::norm(n);
						cv::Vec3f p = normalBufferPoints[currentIdxY][currentIdxX];
						cv::Vec3f toCamera = camPos - p;
						toCamera = toCamera / cv::norm(toCamera);
						float d = n.dot(toCamera);
						if (d < 0) {
							n = -n;
							d = -d;
						}

						if (d > 0.3) {
							out.emplace_back(p, n, 1);
						}
					}
				}
			}
		}
	}
};
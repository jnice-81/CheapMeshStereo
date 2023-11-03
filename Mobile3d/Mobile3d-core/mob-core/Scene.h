#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include "helpers.h"
#include "View.h"

// Correct?
class VecHash {
public:
	std::size_t operator()(const cv::Vec3i& v) const {
		std::hash<int> hasher;
		std::size_t hashValue = hasher(v[0]);
		hashValue ^= hasher(v[1]) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
		hashValue ^= hasher(v[2]) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
		return hashValue;
	}
};

class ScenePoint {
public:
	cv::Vec3f normal;
	float confidence;
};

class RenderHelper {
public:
	RenderHelper(View& v) {
		cv::Matx44d extrinsics;
		cv::Matx44d intrinsics;
		v.extrinsics.convertTo(extrinsics, CV_64F);
		cv::Mat tmpintr = cv::Mat::zeros(4, 4, CV_64F);
		v.intrinsics.copyTo(tmpintr(cv::Rect(0, 0, 3, 3)));
		tmpintr.at<double>(3, 2) = 1;
		tmpintr.convertTo(intrinsics, CV_64F);
		P = intrinsics * extrinsics;
	}

	inline cv::Vec3f projectPoint(const cv::Vec3f p) const {
		cv::Vec4f h(p[0], p[1], p[2], 1);
		h = P * h;
		float d = h[2];
		h /= h[2];
		cv::Vec3f result(h[0], h[1], d);
		return result;
	}

	cv::Matx44d P;
};

class Scene {
public:
	Scene(double voxelSideLength) {
		this->voxelSideLength = voxelSideLength;
	}

	inline void addPoint(cv::Vec3f point, cv::Vec3f normal, float confidence) {
		cv::Vec3i q = floatToIntVec<int, float, 3>(point / voxelSideLength);
		auto old = surfacePoints.find(q);
		if(old == surfacePoints.end()) {
			ScenePoint s;
			s.normal = normal;
			s.confidence = confidence;
			surfacePoints.insert(std::make_pair(q, s));
		}
		else {
			float oldconfidence = old->second.confidence;
			old->second.confidence = oldconfidence + confidence;
			old->second.normal = (oldconfidence * old->second.normal + confidence * normal) / old->second.confidence;
		}
	}

	inline cv::Vec3f addVoxelCenter(const cv::Vec3f voxel) const {
		const float center = (float)voxelSideLength * 0.5f;
		const cv::Vec3f toCenter(center, center, center);
		return voxel + toCenter;
	}

	inline cv::Vec3f voxelToPoint(const cv::Vec3i voxelIdx) const {
		cv::Vec3f p = voxelIdx;
		return addVoxelCenter(p * voxelSideLength);
	}

	inline cv::Vec3f getCenterOfVoxel(const cv::Vec3f point) const {
		cv::Vec3i q = floatToIntVec<int, float, 3>(point / voxelSideLength);
		return addVoxelCenter((cv::Vec3f)q * voxelSideLength);
	}

	inline cv::Vec3f centerAndRenderBack(const RenderHelper& renderHelper, const cv::Vec3f point) const {
		return renderHelper.projectPoint(getCenterOfVoxel(point));
	}

	int filterConfidence(const float minConfidence) {
		auto end = surfacePoints.end();
		std::vector<std::_List_iterator<std::_List_val<std::_List_simple_types<std::pair<const cv::Vec3i, ScenePoint>>>>> toRemove;

		for (auto it = surfacePoints.begin(); it != end; it++) {
			if (it->second.confidence <= minConfidence) {
				toRemove.push_back(it);
			}
		}

		for (auto g : toRemove) {
			surfacePoints.erase(g);
		}
	}

	int filterOutliers(const int l1radius, const int minhits) {
		// This thing is dependent on order, cause as outliers are removed, other points
		// that were before not outliers might become outliers. Anyway this is not handled here
		// for the sake of easy code and speed.
		std::vector<cv::Vec3i> toRemove;

		for (auto it = surfacePoints.begin(); it != surfacePoints.end(); it++) {
			cv::Vec3i c = it->first;
			int hits = 0;

			for (int i = -l1radius; i <= l1radius; i++) {
				for (int j = -l1radius; j <= l1radius; j++) {
					for (int k = -l1radius; k <= l1radius; k++) {
						cv::Vec3i h({ c[0] + i, c[1] + j, c[2] + k });
						if (surfacePoints.find(h) != surfacePoints.end()) {
							hits++;
							if (hits >= minhits) {
								goto END_OF_CHECK;
							}
						}
					}
				}
			}

			END_OF_CHECK:
			if (hits < minhits) {
				toRemove.push_back(c);
			}
		}

		for (auto it = toRemove.begin(); it != toRemove.end(); it++) {
			surfacePoints.erase(*it);
		}

		return toRemove.size();
	}

	void export_xyz(std::string path) {
		std::ofstream f(path, std::ios_base::out);
		for (const auto& t : surfacePoints) {
			cv::Vec3f u = t.first;
			cv::Vec3f v = addVoxelCenter(u * voxelSideLength);
			cv::Vec3f n = t.second.normal;
			f << v[0] << " " << v[1] << " " << v[2] << " " << n[0] << " " << n[1] << " " << n[2] << std::endl;
		}
		f.close();
	}

	void import_xyz(std::string path) {
		std::ifstream f(path, std::ios_base::in);
		int idx = 0;
		float current;
		cv::Vec3f v;
		cv::Vec3f n;
		while (f >> current) {
			int local = idx % 6;
			if (local < 3) {
				v[local] = current;
			}
			else {
				n[local % 3] = current;
			}
			if (local == 5) {
				addPoint(v, n, 1.0);
			}

			idx++;
		}
		if (idx % 6 != 0) {
			std::cerr << "Something was wrong when reading the file in import_xyz";
		}
		f.close();
	}

	inline double getVoxelSideLength() {
		return voxelSideLength;
	}

	inline std::unordered_map<cv::Vec3i, ScenePoint, VecHash>& getScenePoints(){
		return surfacePoints;
	}

	cv::Mat directRender(View& v, float zfar = 1.0f, bool renderNormals = false) {
		cv::Size imgsize = v.image.size();
		cv::Mat result;
		cv::Mat zBuffer = cv::Mat::ones(imgsize, CV_32FC3) * zfar;
		if (renderNormals) {
			result = cv::Mat::zeros(imgsize, CV_32FC3);
		}
		else {
			result = cv::Mat::zeros(imgsize, CV_32F);
		}
		 
		RenderHelper rhelper(v);
		auto endSurface = surfacePoints.end();

		for (auto it = surfacePoints.begin(); it != endSurface; it++) {
			cv::Vec3f p = voxelToPoint(it->first);
			cv::Vec3f project = rhelper.projectPoint(p);
			cv::Vec3i pdash = floatToIntVec<int, float, 3>(project);
			if (pdash[0] >= 0 && pdash[1] >= 0 && project[2] > 0 && pdash[0] < imgsize.width && pdash[1] < imgsize.height) {
				if (renderNormals) {
					cv::Vec3f n = it->second.normal;
					n = (n + vecOnes<cv::Vec3f>() * 1.5) * (1 / 3.0);
					if (zBuffer.at<float>(pdash[1], pdash[0]) >= project[2]) {
						zBuffer.at<float>(pdash[1], pdash[0]) = project[2];
						result.at<cv::Vec3f>(pdash[1], pdash[0]) = n;
					}
				}
				else {
					if (zBuffer.at<float>(pdash[1], pdash[0]) >= project[2]) {
						zBuffer.at<float>(pdash[1], pdash[0]) = project[2];
						result.at<float>(pdash[1], pdash[0]) = project[2] / zfar;
					}
				}
			}
		}

		return result;
	}
private:
	template<typename It, typename Ft, unsigned int Dim>
	static inline cv::Vec<It, Dim> floatToIntVec(const cv::Vec<Ft, Dim> in) {
		cv::Vec<It, Dim> g;
		for (int i = 0; i < Dim; i++) {
			if (in[i] < 0) {
				g[i] = (It)(in[i] - 1);
			}
			else {
				g[i] = (It)in[i];
			}
		}
		return g;
	}

	double voxelSideLength;
	std::unordered_map<cv::Vec3i, ScenePoint, VecHash> surfacePoints;
};
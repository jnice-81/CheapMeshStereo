#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "HierarchicalVoxelGrid.h"
#include "Scene.h"

constexpr int OnLevel = 2;
constexpr int Levels = 3;
typedef Scene<Levels, bool> SceneType;


class SurfaceVoxel {
public:
	cv::Vec3f pos;
	bool faces[3];
};

class SurfaceReconstruct {
	//template<int OnLevel>
	void findNeighborsFor(const cv::Vec3i c, const int l1radius, SceneType &scene, std::vector<SceneType::TreeIterator<OnLevel, Levels>>& out) const {
		for (int i = -l1radius; i <= l1radius; i++) {
			for (int j = -l1radius; j <= l1radius; j++) {
				for (int k = -l1radius; k <= l1radius; k++) {
					cv::Vec3f h = scene.retrievePoint(cv::Vec3i({ c[0] + i, c[1] + j, c[2] + k }), OnLevel);
					auto m = scene.surfacePoints.template findVoxel<OnLevel>(h);

					if (!m.isEnd()) {
						out.push_back(m);
					}
				}
			}
		}
	}

	inline float lininp(float x1, float x2, float y1, float y2, float t) const {
		return y1 + ((y2 - y1) / (x2 - x1)) * (t - x1);
	}

	//template<int OnLevel>
	std::pair<double, double> computeImplicitValue(const cv::Vec3f& p, double s, SceneType& scene) const {
		std::vector<SceneType::TreeIterator<OnLevel, Levels>> neighbors;
		double sidelength = scene.retrieveVoxelSidelength(OnLevel);
		int num_scan = (int)(s / sidelength) + 1;
		findNeighborsFor(scene.retrieveVoxel(p, OnLevel), num_scan, scene, neighbors);

		double weightSum = 0;
		double weightedValueSum = 0;

		for (auto& it : neighbors) {
			while (!it.isEnd()) {
				ScenePoint g = it->second;

				cv::Vec3f diff = g.position - p;
				double dist = cv::norm(diff);
				double x = g.normal.dot(diff);
				double weight = lininp(0.0, s, 1.0, 0.0, dist);
				
				weightedValueSum += x * weight;
				weightSum += weight;
			}
		}

		return std::make_pair(weightedValueSum, weightSum);
	}

	cv::Vec3d computeImplicitNormal(const cv::Vec3f& p, double s, SceneType& scene) const {
		std::vector<SceneType::TreeIterator<OnLevel, Levels>> neighbors;
		double sidelength = scene.retrieveVoxelSidelength(OnLevel);
		int num_scan = (int)(s / sidelength) + 1;
		findNeighborsFor(scene.retrieveVoxel(p, OnLevel), num_scan, scene, neighbors);

		cv::Vec3f result;

		// z(p) = 2norm(g - p)
		// u(p) = g - p

		double sumZp = 0;
		double sumNuz = 0;
		cv::Vec3d sumZd = cv::Vec3d::zeros();
		cv::Vec3d sumG = cv::Vec3d::zeros();

		for (auto& it : neighbors) {
			while (!it.isEnd()) {
				ScenePoint g = it->second;

				cv::Vec3d n = g.normal;
				cv::Vec3d diff = g.position - p;
				double z = cv::norm(diff);
				cv::Vec3d zd = -(2.0 / z) * diff;
				sumZp += z;
				sumG += n * z + n.dot(diff) * zd;
				sumNuz += n.dot(diff) * z;
				sumZd += zd;
			}
		}

		result = (1.0 / sumZp) * sumG + sumNuz * (1.0 / (sumZp * sumZp)) * sumZd;
		return result;
	}
public:
	HierachicalVoxelGrid<1, bool, SurfaceVoxel> svoxel;

	SurfaceReconstruct() : svoxel(0.05, std::vector<int>({ 5 }))
	{
		
	}

	void computeSurface(cv::Vec3i voxel, SceneType& scene) {

		SurfaceVoxel result;

		std::pair<double, double> implicitVals[2][2][2];
		cv::Vec3f zeroPoint = scene.retrievePoint(voxel, 1);
		double sidelength = scene.retrieveVoxelSidelength(1);
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					cv::Vec3f p = cv::Vec3f(x * sidelength, y * sidelength, z * sidelength) + zeroPoint;
					implicitVals[x][y][z] = computeImplicitValue(p, 1.5 * sidelength, scene);
				}
			}
		}

		std::vector<cv::Vec3f> changePoints;
		changePoints.reserve(12);

		float minweight = 0;

		for (int j = 0; j < 3; j++) {
			for (int g1 = 0; g1 < 2; g1++) {
				for (int g2 = 0; g2 < 2; g2++) {
					std::pair<double, double> ck1;
					std::pair<double, double> ck2;
					switch (j) {
					case 0:
						ck1 = implicitVals[0][g1][g2];
						ck2 = implicitVals[1][g1][g2];
						break;
					case 1:
						ck1 = implicitVals[g1][0][g2];
						ck2 = implicitVals[g1][1][g2];
						break;
					case 2:
						ck1 = implicitVals[g1][g2][0];
						ck2 = implicitVals[g1][g2][1];
						break;
					}

					if (ck1.first > 0 != ck2.first > 0 && ck1.second > minweight && ck2.second > minweight) {
						if (g1 == 0 && g2 == 0) {
							result.faces[j] = true;
						}

						cv::Vec3f changepoint;

					}

				}
			}
		}
	}
	}
};

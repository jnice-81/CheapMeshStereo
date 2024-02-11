#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "HierarchicalVoxelGrid.h"
#include "Scene.h"

constexpr int OnLevel = 2;
constexpr int Levels = 3;
const int CornerCacheSize = 2000;
typedef Scene<Levels, bool> SceneType;

/*
No optim: 45s
iterative tracing: 14s (+ quality improved ;)

*/

class SurfaceVoxel {
public:
	SurfaceVoxel()
	{
		for (int i = 0; i < 3; i++) {
			faces[i] = false;
		}
	}

	cv::Vec3f pos;
	bool faces[3];
};

template<typename T, int size>
class CopyArray {
public:
	T r[size];

	CopyArray(T* v) {
		memcpy(r, v, sizeof(T) * size);
	}

	CopyArray() {

	}

	inline T& operator[](std::size_t idx) {
		return r[idx];
	}
};

class SurfaceReconstruct {
private:
	std::unordered_map<cv::Vec3i, float, VecHash> cornerCache;

	/*
	inline void loadCache(const cv::Vec3i &voxel, std::unordered_map<cv::Vec3i, float, VecHash> &corners) const {
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					cornerCache
				}
			}
		}
	}
	*/

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

				if (dist > s) {
					it++;
					continue;
				}

				double x = g.normal.dot(diff);
				double weight = lininp(0.0, s, 1.0, 0.0, dist);
				
				weightedValueSum += x * weight;
				weightSum += weight;

				it++;
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

		double sumZp = 0.001;
		double sumNuz = 0;
		cv::Vec3d sumZd = cv::Vec3d::zeros();
		cv::Vec3d sumG = cv::Vec3d::zeros();

		const float eps = 0.001;

		for (auto& it : neighbors) {
			while (!it.isEnd()) {
				ScenePoint g = it->second;

				cv::Vec3d n = g.normal;
				cv::Vec3d diff = g.position - p;
				double z = cv::norm(diff);

				if (z > s) {
					it++;
					continue;
				}

				cv::Vec3d zd = -(2.0 / (z + eps)) * diff;
				sumZp += z;
				sumG += n * z + n.dot(diff) * zd;
				sumNuz += n.dot(diff) * z;
				sumZd += zd;

				it++;
			}
		}

		result = (1.0 / (sumZp + eps)) * sumG + sumNuz * (1.0 / (sumZp * sumZp + eps)) * sumZd;
		return cv::normalize(result);
	}

	double linearAdapt(double v1, double v2) {
		return -v1 / (v2 - v1);
	}

	inline void computeSurfaceFor(const cv::Vec3i voxel, SceneType& scene, std::unordered_set<cv::Vec3i, VecHash> &foundneighbors) {

		SurfaceVoxel result;
		float sidelength = svoxel.retrieveVoxelSidelength(1);
		const float scale = scalefactor * scene.retrieveVoxelSidelength(Levels);

		std::pair<float, float> implicitVals[2][2][2];
		cv::Vec3f zeroPoint = svoxel.retrieveCornerPoint(voxel, 1);
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					cv::Vec3f p = cv::Vec3f(x * sidelength, y * sidelength, z * sidelength) + zeroPoint;
					implicitVals[x][y][z] = computeImplicitValue(p, scale, scene);
				}
			}
		}

		//exportImplicitVals(implicitVals, zeroPoint, sidelength);

		std::vector<cv::Vec3f> changePoints;
		changePoints.reserve(12);
		cv::Vec3f xVec = cv::Vec3f(sidelength, 0.0, 0.0);
		cv::Vec3f yVec = cv::Vec3f(0.0, sidelength, 0.0);
		cv::Vec3f zVec = cv::Vec3f(0.0, 0.0, sidelength);

		for (int j = 0; j < 3; j++) {
			for (int g1 = 0; g1 < 2; g1++) {
				for (int g2 = 0; g2 < 2; g2++) {
					std::pair<float, float> ck1;
					std::pair<float, float> ck2;
					cv::Vec3f basepoint;
					cv::Vec3f edgevec;

					switch (j) {
					case 0:
						ck1 = implicitVals[0][g1][g2];
						ck2 = implicitVals[1][g1][g2];
						basepoint = zeroPoint + yVec * g1 + zVec * g2;
						edgevec = xVec;
						break;
					case 1:
						ck1 = implicitVals[g1][0][g2];
						ck2 = implicitVals[g1][1][g2];
						basepoint = zeroPoint + xVec * g1 + zVec * g2;
						edgevec = yVec;
						break;
					case 2:
						ck1 = implicitVals[g1][g2][0];
						ck2 = implicitVals[g1][g2][1];
						basepoint = zeroPoint + xVec * g1 + yVec * g2;
						edgevec = zVec;
						break;
					}


					if ((ck1.first >= 0 != ck2.first >= 0 || ck1.first == 0 && ck2.first != 0 || ck1.first != 0 && ck2.first == 0) &&
						ck1.second > minweight && ck2.second > minweight) {
						if (g1 == 0 && g2 == 0) {
							result.faces[j] = true;
						}

						changePoints.push_back(basepoint + linearAdapt(ck1.first, ck2.first) * edgevec);

						cv::Vec3f neighborIterbase = basepoint + 0.5 * edgevec;
						float sl = svoxel.retrieveVoxelSidelength(1);
						for (int m = 0; m < 2; m++) {
							for (int k = 0; k < 2; k++) {
								cv::Vec3f q;
								switch (j)
								{
								case 0:
									q = neighborIterbase + cv::Vec3f(0, (-0.5 + m) * sl, (-0.5 + k) * sl);
									break;
								case 1:
									q = neighborIterbase + cv::Vec3f((-0.5 + m) * sl, 0, (-0.5 + k) * sl);
									break;
								case 2:
									q = neighborIterbase + cv::Vec3f((-0.5 + m) * sl, (-0.5 + k) * sl, 0);
									break;
								}
								foundneighbors.insert(svoxel.retrieveVoxel(q, 1));
							}
						}
						
					}

				}
			}
		}

		if (changePoints.size() == 0) {
			return;
		}
		else {
			foundneighbors.erase(voxel);
		}
		

		int matrixSize = changePoints.size() + 3;
		cv::Mat A = cv::Mat(matrixSize, 3, CV_32F);
		cv::Mat b = cv::Mat(matrixSize, 1, CV_32F);

		for (int i = 0; i < changePoints.size(); i++) {
			cv::Vec3f n = computeImplicitNormal(changePoints[i], scale, scene);

			//DEBUG
			exportImplNorm.addPoint(ScenePoint(changePoints[i], n, 1));

			memcpy(((float*)A.data) + i * 3, n.val, 3 * sizeof(float));
			b.at<float>(i, 0) = n.dot(changePoints[i]);
		}


		cv::Vec3f meanPos = cv::Vec3f::zeros();
		for (int i = 0; i < changePoints.size(); i++) {
			meanPos += changePoints[i];
		}
		meanPos = meanPos * (1.0 / changePoints.size());

		float bias = 1.0;
		cv::Vec3f nX = cv::Vec3f(bias, 0, 0);
		cv::Vec3f nY = cv::Vec3f(0, bias, 0);
		cv::Vec3f nZ = cv::Vec3f(0, 0, bias);
		memcpy(((float*)A.data) + changePoints.size() * 3, nX.val, 3 * sizeof(float));
		memcpy(((float*)A.data) + changePoints.size() * 3 + 3, nY.val, 3 * sizeof(float));
		memcpy(((float*)A.data) + changePoints.size() * 3 + 6, nZ.val, 3 * sizeof(float));
		b.at<float>(changePoints.size(), 0) = nX.dot(meanPos);
		b.at<float>(changePoints.size() + 1, 0) = nY.dot(meanPos);
		b.at<float>(changePoints.size() + 2, 0) = nZ.dot(meanPos);

		//std::cout << A << std::endl << b;

		cv::Vec3f point;
		if (cv::solve(A, b, point, cv::DECOMP_NORMAL)) {
			result.pos = point;
		}
		else {
			result.pos = svoxel.retrieveVoxelCenter(1) + zeroPoint;
		}

		cv::Vec3f insert_point = svoxel.retrievePoint(voxel, 1);
		svoxel.surfacePoints.insert_or_update(insert_point, result);
	}

public:
	float minweight;
	float scalefactor;
	HierachicalVoxelGrid<1, bool, SurfaceVoxel> svoxel;

	SurfaceReconstruct(float sidelength, float minweight = 20.0, float scalefactor = 3.0) :
		minweight(minweight), scalefactor(scalefactor), svoxel(sidelength, std::vector<int>({ 5 })), exportImplNorm(0.001, std::vector<int>({5}))
	{
		cornerCache.reserve(CornerCacheSize);
	}

	void exportImplicitVals(std::pair<float, float> implicitVals[2][2][2], cv::Vec3f zeroPoint, float sidelength) {
		Scene<1, ScenePoint> e(0.001, std::vector<int>({ 5 }));
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					cv::Vec3f p = cv::Vec3f(x * sidelength, y * sidelength, z * sidelength) + zeroPoint;
					e.addPoint(ScenePoint(p, cv::Vec3f(0, 1.0 * implicitVals[x][y][z].first), 1));
				}
			}
		}
		e.export_xyz("impl.xyz");
	}

	Scene<1, ScenePoint> exportImplNorm;

	void computeSurface(SceneType& scene) {
		std::unordered_set<cv::Vec3i, VecHash> currentComputationVoxels, futureComputationVoxels, pastComputationVoxels, currentOutput;

		for (auto it = scene.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			currentComputationVoxels.insert(svoxel.retrieveVoxel(it->second.position, 1));
		}
		

		while (currentComputationVoxels.size() > 0) {
			int cc = 0;
			for (const auto& v : currentComputationVoxels) {
				cc++;
				computeSurfaceFor(v, scene, currentOutput);

				for (const auto& n : currentOutput) {
					if (currentComputationVoxels.find(n) == currentComputationVoxels.end() && pastComputationVoxels.find(n) == pastComputationVoxels.end()) {
						futureComputationVoxels.insert(n);
					}
				}

				currentOutput.clear();
			}

			std::cout << "Round done " << futureComputationVoxels.size() << std::endl;
			
			pastComputationVoxels.insert(currentComputationVoxels.begin(), currentComputationVoxels.end());
			currentComputationVoxels.clear();
			currentComputationVoxels.swap(futureComputationVoxels);
		}

	}

	inline std::pair<std::size_t, bool> allocateVertexForVoxel(std::vector<CopyArray<float, 3>>& vertices, HierachicalVoxelGrid<1, bool, int> &vPos, const cv::Vec3i &voxel) {
		int result;
		cv::Vec3f middle = svoxel.retrievePoint(voxel, 1);
		auto it = vPos.surfacePoints.findVoxel<1>(middle);
		if (it.isEnd()) {
			auto svoxelptr = svoxel.surfacePoints.findVoxel<1>(middle);
			if (svoxelptr.isEnd()) {
				return std::make_pair(0, false);
			}
			cv::Vec3f p = svoxelptr->second.pos;
			vertices.push_back(CopyArray<float, 3>(p.val));
			result = vertices.size() - 1;
			vPos.surfacePoints.insert_or_update(middle, result);
		}
		else {
			result = it->second;
		}
		return std::make_pair(result, true);
	}

	inline void exportFace(std::vector<CopyArray<float, 3>>& vertices, HierachicalVoxelGrid<1, bool, int> &vPos, std::vector<CopyArray<int, 6>>& faces, 
		const cv::Vec3i& base, const std::vector<cv::Vec3i>& relativeIncludes) {

		int result[6];
		for (int i = 0; i < 4; i++) {
			const cv::Vec3i& a = relativeIncludes[i];
			cv::Vec3i current = base + a;
			std::pair<int, bool> alloc = allocateVertexForVoxel(vertices, vPos, current);
			if (!alloc.second) {
				return;
			}

			int currentPos = alloc.first;
			switch (i)
			{
			case 0:
				result[0] = currentPos;
				break;
			case 1:
				result[1] = currentPos;
				result[3] = currentPos;
				break;
			case 2:
				result[2] = currentPos;
				result[4] = currentPos;
				break;
			case 3:
				result[5] = currentPos;
				break;
			}
		}

		faces.push_back(result);
	}

	void toConnectedMesh(std::vector<CopyArray<float, 3>>& vertices, std::vector<CopyArray<int, 6>>& faces) {
		HierachicalVoxelGrid<1, bool, int> vPos(svoxel.retrieveVoxelSidelength(1), std::vector<int>({ 5 }));
		std::vector<std::vector<cv::Vec3i>> relativeNeighbors = { 
			{cv::Vec3i(0, 0, 0), cv::Vec3i(0, -1, 0), cv::Vec3i(0, 0, -1), cv::Vec3i(0, -1, -1)},
			{cv::Vec3i(0, 0, 0), cv::Vec3i(-1, 0, 0), cv::Vec3i(0, 0, -1), cv::Vec3i(-1, 0, -1)},
			{cv::Vec3i(0, 0, 0), cv::Vec3i(-1, 0, 0), cv::Vec3i(0, -1, 0), cv::Vec3i(-1, -1, 0)}};

		auto it = svoxel.surfacePoints.treeIteratorBegin();
		while (!it.isEnd()) {

			const auto& c = it->second;

			for (int i = 0; i < 3; i++) {
				if (c.faces[i]) {
					exportFace(vertices, vPos, faces, it->first, relativeNeighbors[i]);
				}
			}

			it++;
		}
	}

	void exportObj(std::string path) {
		std::vector<CopyArray<float, 3>> vertices;
		std::vector<CopyArray<int, 6>> faces;

		toConnectedMesh(vertices, faces);

		std::ofstream g(path, std::ios_base::out);

		for (CopyArray<float, 3>& f : vertices) {
			g << "v " << f[0] << " " << f[1] << " " << f[2] << std::endl;
		}

		for (auto& f : faces) {
			g << "f " << f[0]+1 << " " << f[1]+1 << " " << f[2]+1 << std::endl;
			g << "f " << f[3]+1 << " " << f[4]+1 << " " << f[5]+1 << std::endl;
		}

		g.close();
	}
};

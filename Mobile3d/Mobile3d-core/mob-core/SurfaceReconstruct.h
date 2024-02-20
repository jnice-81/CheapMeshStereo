#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "HierarchicalVoxelGrid.h"
#include "Scene.h"

constexpr int OnLevel = 2;
constexpr int Levels = 3;
const int CornerCacheSize = 30000;
const int NormalsCacheSize = 5000;
typedef Scene<Levels> SceneType;

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
	struct VecPairHasher {
		std::size_t operator()(const std::pair<cv::Vec3i, cv::Vec3i>& p) const {
			VecHash vh;
			std::size_t hash1 = vh(p.first);
			std::size_t hash2 = vh(p.second);

			return hash1 ^ (hash2 << 1);
		}
	};

	std::unordered_map<cv::Vec3i, std::pair<float, float>, VecHash> cornerCache;
	std::unordered_map<std::pair<cv::Vec3i, cv::Vec3i>, cv::Vec3f, VecPairHasher> normalsCache;
	float minweight;
	float scalefactor;
	HierachicalVoxelGrid<1, SurfaceVoxel> svoxel;

	/*
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
	*/

	inline void loadCache(const cv::Vec3i &voxel, bool implicitCacheValid[2][2][2], std::pair<float, float> implicitVals[2][2][2]) const {
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					auto it = cornerCache.find(voxel + cv::Vec3i(x, y, z));
					if (it != cornerCache.end()) {
						implicitCacheValid[x][y][z] = true;
						implicitVals[x][y][z] = it->second;
					}
					else {
						implicitCacheValid[x][y][z] = false;
					}
				}
			}
		}
	}

	inline void storeCache(const cv::Vec3i& voxel, std::pair<float, float> implicitVals[2][2][2]) {
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					cornerCache.insert(std::make_pair(voxel + cv::Vec3i(x, y, z), implicitVals[x][y][z]));
				}
			}
		}
		while (cornerCache.size() > CornerCacheSize) {
			cornerCache.erase(cornerCache.begin());
		}
	}

	inline void loadNormalCache(const std::vector<std::pair<cv::Vec3i, cv::Vec3i>>& indizes, std::vector<std::pair<cv::Vec3f, bool>>& out) {
		for (const auto& idx : indizes) {
			auto it = normalsCache.find(idx);
			if (it == normalsCache.end()) {
				out.push_back(std::make_pair(cv::Vec3f(), false));
			}
			else {
				out.push_back(std::make_pair(it->second, true));
			}
		}
	}

	inline void storeNormalCache(const std::vector<std::pair<cv::Vec3i, cv::Vec3i>>& indizes, const std::vector<std::pair<cv::Vec3f, bool>>& store) {
		for (std::size_t i = 0; i < indizes.size(); i++) {
			if (!store[i].second) {
				normalsCache.insert(std::make_pair(indizes[i], store[i].first));
			}
		}
		while (normalsCache.size() > NormalsCacheSize) {
			normalsCache.erase(normalsCache.begin());
		}
	}

	void findNeighborsFor(const cv::Vec3f p, const float l1radius, SceneType& scene, std::vector<SceneType::TreeIterator<OnLevel, Levels>>& out) const {
		cv::Vec3f l1radi;
		cv::Vec3f cornerQuad;

		auto tmp = cv::Vec3f(l1radius, l1radius, l1radius);
		cornerQuad = p - tmp;
		l1radi = 2 * tmp;
		float voxelSidelength = scene.retrieveVoxelSidelength(OnLevel);
		const float numStabilityMargin = 0.1;
		float numericalStabilityLength = numStabilityMargin * voxelSidelength;

		cv::Vec3f t = cornerQuad / voxelSidelength;
		for (int i = 0; i < 3; i++) {
			float divInt = std::abs(t[i] - roundf(t[i]));
			if (divInt < numStabilityMargin) {
				cornerQuad[i] -= numericalStabilityLength;
				l1radi[i] += 2 * numericalStabilityLength;
			}
		}

		std::unordered_set<cv::Vec3i, VecHash> select;

		for (int x = 0; x <= (int)(l1radi[0] / voxelSidelength) + 1; x++) {
			for (int y = 0; y <= (int)(l1radi[1] / voxelSidelength) + 1; y++) {
				for (int z = 0; z <= (int)(l1radi[2] / voxelSidelength) + 1; z++) {
					cv::Vec3f q = cv::Vec3f(
						std::clamp(x * voxelSidelength, 0.0f, l1radi[0]),
						std::clamp(y * voxelSidelength, 0.0f, l1radi[1]),
						std::clamp(z * voxelSidelength, 0.0f, l1radi[2])) + cornerQuad;
					select.insert(scene.retrieveVoxel(q, OnLevel));
				}
			}
		}

		out.reserve(select.size());
		for (const auto& g : select) {
			auto it = scene.surfacePoints.findVoxel<OnLevel>(scene.retrievePoint(g, OnLevel));
			if (!it.isEnd()) {
				out.push_back(it);
			}
		}
	}

	inline float lininp(float x1, float x2, float y1, float y2, float t) const {
		return y1 + ((y2 - y1) / (x2 - x1)) * (t - x1);
	}

	//template<int OnLevel>
	std::pair<double, double> computeImplicitValue(const cv::Vec3f& p, double s, SceneType& scene) const {
		std::vector<SceneType::TreeIterator<OnLevel, Levels>> neighbors;
		findNeighborsFor(p, s, scene, neighbors);

		double weightSum = 0;
		double weightedValueSum = 0;

		for (auto& it : neighbors) {
			int count = it.getLevelInfo<OnLevel>().pointCount;
			for (int i = 0; i < count; i++) {
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
		findNeighborsFor(p, s, scene, neighbors);

		cv::Vec3f result;

		// z(p) = 2norm(g - p)
		// u(p) = g - p

		double sumZp = 0.001;
		double sumNuz = 0;
		cv::Vec3d sumZd = cv::Vec3d::zeros();
		cv::Vec3d sumG = cv::Vec3d::zeros();

		const float eps = 0.001;

		for (auto& it : neighbors) {
			int count = it.getLevelInfo<OnLevel>().pointCount;
			for (int i = 0; i < count; i++) {
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

		bool implicitCacheValid[2][2][2];
		std::pair<float, float> implicitVals[2][2][2];
		
		loadCache(voxel, implicitCacheValid, implicitVals);

		cv::Vec3f zeroPoint = svoxel.retrieveCornerPoint(voxel, 1);
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					if (!implicitCacheValid[x][y][z]) {
						cv::Vec3f p = cv::Vec3f(x * sidelength, y * sidelength, z * sidelength) + zeroPoint;
						implicitVals[x][y][z] = computeImplicitValue(p, scale, scene);
					}
				}
			}
		}

		storeCache(voxel, implicitVals);

		std::vector<cv::Vec3f> changePoints;
		changePoints.reserve(12);
		std::vector<std::pair<cv::Vec3i, cv::Vec3i>> normalCacheIdx;
		normalCacheIdx.reserve(12);
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
					cv::Vec3i p1, p2;

					switch (j) {
					case 0:
						ck1 = implicitVals[0][g1][g2];
						ck2 = implicitVals[1][g1][g2];
						p1 = cv::Vec3i(0, g1, g2);
						p2 = cv::Vec3i(1, g1, g2);
						basepoint = zeroPoint + yVec * g1 + zVec * g2;
						edgevec = xVec;
						break;
					case 1:
						ck1 = implicitVals[g1][0][g2];
						ck2 = implicitVals[g1][1][g2];
						p1 = cv::Vec3i(g1, 0, g2);
						p2 = cv::Vec3i(g1, 1, g2);
						basepoint = zeroPoint + xVec * g1 + zVec * g2;
						edgevec = yVec;
						break;
					case 2:
						ck1 = implicitVals[g1][g2][0];
						ck2 = implicitVals[g1][g2][1];
						p1 = cv::Vec3i(g1, g2, 0);
						p2 = cv::Vec3i(g1, g2, 1);
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
						normalCacheIdx.push_back(std::make_pair(voxel + p1, voxel + p2));

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
		
		std::vector<std::pair<cv::Vec3f, bool>> loadedNormalCache;
		loadedNormalCache.reserve(changePoints.size());
		loadNormalCache(normalCacheIdx, loadedNormalCache);

		int matrixSize = changePoints.size() + 3;
		cv::Mat A = cv::Mat(matrixSize, 3, CV_32F);
		cv::Mat b = cv::Mat(matrixSize, 1, CV_32F);

		for (int i = 0; i < changePoints.size(); i++) {
			cv::Vec3f n;
			if (loadedNormalCache[i].second) {
				n = loadedNormalCache[i].first;
			}
			else {
				n = computeImplicitNormal(changePoints[i], scale, scene);
				loadedNormalCache[i].first = n;
			}

			memcpy(((float*)A.data) + i * 3, n.val, 3 * sizeof(float));
			b.at<float>(i, 0) = n.dot(changePoints[i]);
		}

		storeNormalCache(normalCacheIdx, loadedNormalCache);


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

	SurfaceReconstruct(float sidelength, float minweight = 20.0, float scalefactor = 3.0) :
		minweight(minweight), scalefactor(scalefactor), svoxel(sidelength, std::vector<int>({ 5 }))
	{
		cornerCache.reserve(CornerCacheSize + 8);
		normalsCache.reserve(NormalsCacheSize + 12);
	}

	void computeSurface(SceneType& scene) {
		std::unordered_set<cv::Vec3i, VecHash> futureComputationVoxels, pastComputationVoxels, currentOutput;
		constexpr int CacheLocalityLevels = 2;
		HierachicalVoxelGrid<CacheLocalityLevels, bool> currentComputationVoxels(svoxel.retrieveVoxelSidelength(1), std::vector<int>({ 5, 5 }));

		for (auto it = scene.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			currentComputationVoxels.surfacePoints.insert_or_update(it->second.position, false);
		}

		/*
		const float scale = 0.05;
		for (auto it = currentComputationVoxels.begin(); it != currentComputationVoxels.end(); it++) {
			cv::Vec3f u = svoxel.retrievePoint(*it, 1);
			std::vector<SceneType::TreeIterator<OnLevel, Levels>> out;

			cv::Vec3i g = scene.retrieveVoxel(u, OnLevel);

			findNeighborsFor(u, scale, scene, out);
			int closePoints = 0, farPoints = 0;
			for (auto it1 = out.begin(); it1 != out.end(); it1++) {
				auto it2 = *it1;
				auto info = it2.getLevelInfo<OnLevel>();
				for (int i = 0; i < info.pointCount; i++) {
					float dist = cv::norm(u - it2->second.position);
					if (dist > scale) {
						farPoints++;
					}
					else {
						closePoints++;
					}
					it2++;
				}
			}
			std::cout << "Close " << closePoints << " Far " << farPoints << " Total " << closePoints + farPoints << std::endl;

			auto sit = scene.surfacePoints.treeIteratorBegin();
			farPoints = 0; closePoints = 0;
			while (!sit.isEnd()) {
				if (cv::norm(u - sit->second.position) > scale) {
					farPoints++;
				}
				else {
					closePoints++;
				}
				sit++;
			}
			std::cout << "Actual :" << "Close " << closePoints << " Far " << farPoints << " Total " << closePoints + farPoints << std::endl;

		}
		*/

		while (currentComputationVoxels.surfacePoints.getPointCount() > 0) {
			
			for (auto it = currentComputationVoxels.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
				computeSurfaceFor(it->first, scene, currentOutput);

				for (const auto& n : currentOutput) {
					if (currentComputationVoxels.surfacePoints.findVoxel<CacheLocalityLevels>(svoxel.retrievePoint(n, 1)).isEnd() 
							&& pastComputationVoxels.find(n) == pastComputationVoxels.end()) {
						futureComputationVoxels.insert(n);
					}
				}

				currentOutput.clear();
			}
			
			pastComputationVoxels.reserve(pastComputationVoxels.size() + currentComputationVoxels.surfacePoints.getPointCount());
			for (auto it = currentComputationVoxels.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
				pastComputationVoxels.insert(it->first);
			}
			currentComputationVoxels.surfacePoints.clear();
			for (const auto& g : futureComputationVoxels) {
				currentComputationVoxels.surfacePoints.insert_or_update(currentComputationVoxels.retrievePoint(g, CacheLocalityLevels), false);
			}
			futureComputationVoxels.clear();
		}

	}

	inline std::pair<std::size_t, bool> allocateVertexForVoxel(std::vector<CopyArray<float, 3>>& vertices, HierachicalVoxelGrid<1, int> &vPos, const cv::Vec3i &voxel) {
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

	inline void exportFace(std::vector<CopyArray<float, 3>>& vertices, HierachicalVoxelGrid<1, int> &vPos, std::vector<CopyArray<int, 6>>& faces, 
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
		HierachicalVoxelGrid<1, int> vPos(svoxel.retrieveVoxelSidelength(1), std::vector<int>({ 5 }));
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

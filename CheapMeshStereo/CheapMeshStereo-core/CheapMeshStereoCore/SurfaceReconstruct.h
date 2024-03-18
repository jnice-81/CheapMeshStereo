#pragma once

#include "Scene.h"

/*
An array that can be copied.
*/
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

/*
Class for generating meshes from oriented pointclouds (stored as Scene type).
OnLevel: On Which level of the hierarchy of the Scene to search for closeby points. 
	Ideally the voxel sidelenght on that level should be slightly larger than scale (explained on constructor)
	times the lowest level voxel size of the scene. (E.g. use the second lowest level for extraction, set its size to 4 if you set the scale to e.g. 3.5)
Levels: How many levels the Scene has
*/
template<int OnLevel, int Levels>
class SurfaceReconstruct {
public:
	typedef Scene<Levels> SceneType;

private:
	/*
	A class storing the surfaces inside of a single voxel. 
	pos: The position of the vertex inside the voxel.
	faces: Three booleans storing if there are faces to neighbouring voxels.
		There are at most 12 faces that could be connected to a single voxel - however if there is
		a face or not is stored by each voxel in a way that ensures that the presence of a face is only
		stored once. (If Voxel A has a face to voxel B, C, D then only one of them will store that face;
		This is why one gets away with storing only three faces)
	*/
	class SurfaceVoxel {
	public:
		SurfaceVoxel()
		{
			boolstorage = 0;
		}

		cv::Vec3f pos;
		char boolstorage;

		inline bool getFace(const int idx) const {
			return (boolstorage & (1 << idx)) != 0;
		}

		inline void setFaceTrue(const int idx) {
			boolstorage = (0b1 << idx) | boolstorage;
		}
	};

	/*
	Simple hash function for pairs of cv::Vec3i
	*/
	struct VecPairHasher {
		std::size_t operator()(const std::pair<cv::Vec3i, cv::Vec3i>& p) const {
			VecHash vh;
			std::size_t hash1 = vh(p.first);
			std::size_t hash2 = vh(p.second);

			return hash1 ^ (hash2 << 1);
		}
	};

	// The caches storing the implicit value of the function for particular corners respectively edges where a face crosses
	std::unordered_map<cv::Vec3i, std::pair<float, float>, VecHash> cornerCache;
	std::unordered_map<std::pair<cv::Vec3i, cv::Vec3i>, cv::Vec3f, VecPairHasher> normalsCache;

	float minweight;
	float scalefactor;
	float bias;
	int CornerCacheSize = 30000;
	int NormalsCacheSize = 5000;
	// The actual storage of the mesh
	HierachicalVoxelGrid<1, SurfaceVoxel> svoxel;
	SceneType& scene;


	// Loads all corner cache values for a voxel
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

	// Stores all corner cache values for a voxel
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

	// Load all normal cache values for a list of edges
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

	// Store all normal cache values for a list of edges
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

	// Perform linear interpolation
	inline float lininp(float x1, float x2, float y1, float y2, float t) const {
		return y1 + ((y2 - y1) / (x2 - x1)) * (t - x1);
	}

	// Compute the implicit value at p, assuming points have a scale of s
	std::pair<double, double> computeImplicitValue(const cv::Vec3f& p, double s) const {
		auto neighbors = scene.template findNeighborsFor<OnLevel>(p, s);

		double weightSum = 0;
		double weightedValueSum = 0;

		for (auto& it : neighbors) {
			int count = it.template getLevelInfo<OnLevel>().pointCount;
			for (int i = 0; i < count; i++) {
				ScenePoint g = it->second;

				cv::Vec3f diff = g.position - p;
				double dist = cv::norm(diff);

				if (dist > s) {
					it++;
					continue;
				}

				cv::Vec3f n = cv::normalize(g.normal);
				double x = n.dot(diff);
				double weight = lininp(0.0, s, 1.0, 0.0, dist) * g.numhits;
				
				weightedValueSum += x * weight;
				weightSum += weight;

				it++;
			}
		}

		return std::make_pair(weightedValueSum, weightSum);
	}

	// Compute the implicit normal (first derivative of implicit function) at p assuming points have scale s
	cv::Vec3d computeImplicitNormal(const cv::Vec3f& p, double s) const {
		auto neighbors = scene.template findNeighborsFor<OnLevel>(p, s);

		cv::Vec3f result;

		/*
		F(p) = sum(n^T*m(p)) / sum(c * l(d(p)),
		m(p) = u - p
		d(p) = sqrt(m(p)^T * m(p))
		l(x) = 1 - x/s

		=> 
		F'(p) = -( sum(n) * 1 / sum(c * l(d(p))))
				+ sum(n^Tm(p)) * (1 / sum(cl(d(p)))^2) * (sum(c * 1/(sd(p)) * m(p))
		*/

		cv::Vec3f sumN = cv::Vec3f::zeros();
		float sumcldp = 0;
		float sumNmp = 0;
		cv::Vec3f sumc1sdpmp = cv::Vec3f::zeros();

		const float eps = 0.001;
		const float inverseS = 1.0 / s;

		for (auto& it : neighbors) {
			int count = it.template getLevelInfo<OnLevel>().pointCount;
			for (int i = 0; i < count; i++) {
				ScenePoint g = it->second;

				cv::Vec3f diff = g.position - p;
				float z = cv::norm(diff);

				if (z > s) {
					it++;
					continue;
				}

				cv::Vec3f n = cv::normalize(g.normal);
				sumN += n;
				sumcldp += g.numhits * lininp(0.0, s, 1.0, 0.0, z);
				sumNmp += n.dot(diff);
				sumc1sdpmp += g.numhits * (1.0f / (s * z)) * diff;

				it++;
			}
		}

		result = -(sumN * (1.0 / sumcldp)) + sumNmp * (1.0f / (sumcldp * sumcldp)) * sumc1sdpmp;
		result *= -1;
		return cv::normalize(result);
	}

	// Find the point between 0 and 1 where a linear function crosses 0, if it has value v1 at 0 and v2 at 1
	double linearAdapt(double v1, double v2) {
		return -v1 / (v2 - v1);
	}

	// Computes a single SurfaceVoxel at a position
	inline SurfaceVoxel computeSurfaceFor(const cv::Vec3i voxel, std::unordered_set<cv::Vec3i, VecHash> &foundneighbors) {

		SurfaceVoxel result;
		float sidelength = svoxel.retrieveVoxelSidelength(1);
		const float scale = scalefactor * scene.retrieveVoxelSidelength(Levels);

		// Compute/load implicit values
		bool implicitCacheValid[2][2][2];
		std::pair<float, float> implicitVals[2][2][2];
		
		loadCache(voxel, implicitCacheValid, implicitVals);

		cv::Vec3f zeroPoint = svoxel.retrieveCornerPoint(voxel, 1);
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int z = 0; z < 2; z++) {
					if (!implicitCacheValid[x][y][z]) {
						cv::Vec3f p = cv::Vec3f(x * sidelength, y * sidelength, z * sidelength) + zeroPoint;
						implicitVals[x][y][z] = computeImplicitValue(p, scale);
					}
				}
			}
		}

		storeCache(voxel, implicitVals);

		// Determine points on the edges of the voxel where an edge passes trough
		// (Function negative on one side, positive on the other and the weight is large enough at both sides)
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
							result.setFaceTrue(j);
						}

						changePoints.push_back(basepoint + linearAdapt(ck1.first, ck2.first) * edgevec);
						normalCacheIdx.push_back(std::make_pair(voxel + p1, voxel + p2));

						// Add the neighbors to which this voxel has an edge to the foundneighbors
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
			return result;
		}
		
		// Code above will add ourselfs as a neighbor if there is at least one face. This should be prevented.
		foundneighbors.erase(voxel);

		// Find the mean position of all edgecrossings of faces. This is a "good" position for the vertex
		cv::Vec3f meanPos = cv::Vec3f::zeros();
		for (int i = 0; i < changePoints.size(); i++) {
			meanPos += changePoints[i];
		}
		meanPos = meanPos * (1.0 / changePoints.size());
		
		// If there is a positive bias try to compute the vertex position such that it "fits" the normals well.
		// Otherwise simply directly use the meanPos.
		if (bias >= 0) {
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
					n = computeImplicitNormal(changePoints[i], scale);
					loadedNormalCache[i].first = n;
				}

				memcpy(((float*)A.data) + i * 3, n.val, 3 * sizeof(float));
				b.at<float>(i, 0) = n.dot(changePoints[i]);
			}

			storeNormalCache(normalCacheIdx, loadedNormalCache);


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
				result.pos = meanPos;
			}
		}
		else {
			result.pos = meanPos;
		}
		
		return result;
	}

	// Allocate or return a vertex in the list (used during mesh conversion)
	inline std::pair<std::size_t, bool> allocateVertexForVoxel(std::vector<CopyArray<float, 3>>& vertices, HierachicalVoxelGrid<1, int>& vPos, const cv::Vec3i& voxel) {
		int result;
		cv::Vec3f middle = svoxel.retrievePoint(voxel, 1);
		auto it = vPos.surfacePoints.findVoxel<1>(middle);
		if (it.isEnd()) {
			auto svoxelptr = svoxel.surfacePoints.template findVoxel<1>(middle);
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

	// Export a face, adding all vertices to the output list if not already present (Otherwise the existing indices are used)
	inline void exportFace(std::vector<CopyArray<float, 3>>& vertices, HierachicalVoxelGrid<1, int>& vPos, std::vector<CopyArray<int, 6>>& faces,
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

public:

	/*
	Initialize the class.

	sidelength: The voxel sidelength of the Scene with which this is used (lowest level)
	minweight: The minimum weight required for a surface to be extracted. Use low value for clean pointclouds,
		large value if lots of outliers and noise.
	scalefactor: Essentially how big the region is where a point will influence the surface. The scalefactor is multiplied
		by the lowest level voxel sidelenght of the scene. Larger values will result in smoother surfaces with less holes (tendence to oversmoothing),
		and take more time to compute.
	CornerCacheSize: The size of the cache for storing values of the implicit function
	bias: If negative dual countouring is kinda ignored, and the derivative will not influence vertex positions
		Faster to compute, and for noisy data potentially better results.
		If positive the derivative will be taken into account to find a good fit, where a larger bias will again give a tendency
		to converge to the basic point used also when bias is negative. Can give potentially more detail. Especially for higher resolutions
		it actually does not have a large influence but is expensive hence the recommendation there is to use a negative value.
	NormalsCacheSize: The size of the cache used for storing derivatives of the implicit function. If bias negative this cache is not used.
	*/
	SurfaceReconstruct(SceneType &scene, float minweight = 1.0, float scalefactor = 3.0, 
			const int CornerCacheSize = 50000, float bias = -1.0, const int NormalsCacheSize = 0):
		minweight(minweight), scalefactor(scalefactor), scene(scene), svoxel(scene.retrieveVoxelSidelength(Levels), std::vector<int>({ 5 })),
		bias(bias), CornerCacheSize(CornerCacheSize), NormalsCacheSize(NormalsCacheSize)
	{
		cornerCache.reserve(CornerCacheSize + 8);
		normalsCache.reserve(NormalsCacheSize + 12);
	}

	/*
	Computes the entire surface defined by the points in scene. This will only update the internal 
	representation of the scene, and needs to be called prior to surface extraction. 
	It can in principle be called multiple times without affecting the result.
	*/
	void computeSurface() {

		std::vector<cv::Vec3i> dfs_stack;
		std::unordered_set<cv::Vec3i, VecHash> neighbors;

		for (auto it = scene.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			dfs_stack.push_back(it->first);

			while (!dfs_stack.empty()) {
				cv::Vec3i voxel = dfs_stack.back();
				dfs_stack.pop_back();
				cv::Vec3f vpoint = svoxel.retrievePoint(voxel, 1);

				if (svoxel.surfacePoints.findVoxel<1>(vpoint).isEnd()) {
					
					SurfaceVoxel r = computeSurfaceFor(voxel, neighbors);
					svoxel.surfacePoints.insert_or_update(vpoint, r);
					for (const auto& n : neighbors) {
						if (svoxel.surfacePoints.findVoxel<1>(svoxel.retrievePoint(n, 1)).isEnd()) {
							dfs_stack.push_back(n);
						}
					}
					neighbors.clear();

				}
			} 
		}

		/*
		std::unordered_set<cv::Vec3i, VecHash> futureComputationVoxels, pastComputationVoxels, currentOutput;

		// Note: This uses a HierarchicalVoxelGrid to make sure that closeby voxels are computed roughly in order
		// giving large speedups, because it increases cache locality. The bool is actually not used (more a set here)
		constexpr int CacheLocalityLevels = 2;
		HierachicalVoxelGrid<CacheLocalityLevels, bool> currentComputationVoxels(svoxel.retrieveVoxelSidelength(1), std::vector<int>({ 5, 5 }));

		for (auto it = scene.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			currentComputationVoxels.surfacePoints.insert_or_update(it->second.position, false);
		}

		while (currentComputationVoxels.surfacePoints.getPointCount() > 0) {
			
			for (auto it = currentComputationVoxels.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
				SurfaceVoxel r = computeSurfaceFor(it->first, currentOutput);
				svoxel.surfacePoints.insert_or_update(svoxel.retrievePoint(it->first, 1), r);

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
		*/

	}

	/*
	Create a mesh (Needs to be called after computeSurface).
	vertices: A list of CopyArrays (essentially just float array of size 3) storing the vertices
	faces: A list of CopyArrays of length 6 storing the face indices. (6 because each element stores 2 triangles)
		The indices start at 0.

	Note: For both the actual use case is likely to directly access the underlying array, getting a float array where 
		three consecutive floats define a vertex, and 3 consecutive ints define a face.
	*/
	void toConnectedMesh(std::vector<CopyArray<float, 3>>& vertices, std::vector<CopyArray<int, 6>>& faces) {
		HierachicalVoxelGrid<1, int> vPos(svoxel.retrieveVoxelSidelength(1), std::vector<int>({ 5 }));
		std::vector<std::vector<cv::Vec3i>> relativeNeighbors = { 
			{cv::Vec3i(0, 0, 0), cv::Vec3i(0, -1, 0), cv::Vec3i(0, 0, -1), cv::Vec3i(0, -1, -1)},
			{cv::Vec3i(0, 0, 0), cv::Vec3i(-1, 0, 0), cv::Vec3i(0, 0, -1), cv::Vec3i(-1, 0, -1)},
			{cv::Vec3i(0, 0, 0), cv::Vec3i(-1, 0, 0), cv::Vec3i(0, -1, 0), cv::Vec3i(-1, -1, 0)}};

		auto it = svoxel.surfacePoints.treeIteratorBegin();
		while (!it.isEnd()) {

			const SurfaceVoxel& c = it->second;

			for (int i = 0; i < 3; i++) {
				if (c.getFace(i)) {
					exportFace(vertices, vPos, faces, it->first, relativeNeighbors[i]);
				}
			}

			it++;
		}
	}

	/*
	Convenience function that calls toConnectedMesh and exports the mesh as .obj.

	path: The path to which to export
	*/
	void exportObj(std::string path) {
		std::vector<CopyArray<float, 3>> vertices;
		std::vector<CopyArray<int, 6>> faces;

		toConnectedMesh(vertices, faces);

		std::ofstream g(path, std::ios_base::out);

		for (CopyArray<float, 3> &f: vertices) {
			g << "v " << f[0] << " " << f[1] << " " << f[2] << std::endl;
		}

		for (auto &f: faces) {
			g << "f " << f[0] + 1 << " " << f[1] + 1 << " " << f[2] + 1 << std::endl;
			g << "f " << f[3] + 1 << " " << f[4] + 1 << " " << f[5] + 1 << std::endl;
		}

		g.close();
	}
};

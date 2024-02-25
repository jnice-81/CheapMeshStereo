#pragma once

#include <fstream>

#include "HierarchicalVoxelGrid.h"
#include "ScenePoint.h"

/*
A storage for point clouds built on top of HierarchicalVoxelGrid. (Rather: It is a hierarchical voxel grid)
Provides convenience functions, but most of the time using this, one will end up using surfacePoints of HierarchicalVoxelGrid.

Levels: The number of Levels of the hierarchical Grid
*/
template<int Levels>
class Scene : public HierachicalVoxelGrid<Levels, ScenePoint> {
private:
	class NormalPrio {
	public:
		float dist;
		cv::Vec3f pos;

		bool operator<(const NormalPrio& other) const {
			return other.dist < dist;
		}
	};

public:
	Scene(double voxelSideLength, std::vector<int> indexBlocks) : HierachicalVoxelGrid<Levels, ScenePoint>(voxelSideLength, indexBlocks) {

	}

	/*
	Count how many points exist on a given level with a (l1) distance of at most l1radius voxels.

	OnLevel: On which level to count.
	c: The index of the center voxel (the voxel to count for)
	l1radius: An integer specifying the number of voxels to count in each direction
	count_max: Function returns early if value is higher than this. In this case count_max is returned.
	*/
	template<int OnLevel>
	inline std::size_t countNeighborsFor(const cv::Vec3i c, const int l1radius, const int count_max = 0) {
		size_t hits = 0;

		for (int i = -l1radius; i <= l1radius; i++) {
			for (int j = -l1radius; j <= l1radius; j++) {
				for (int k = -l1radius; k <= l1radius; k++) {
					cv::Vec3f h = this->retrievePoint(cv::Vec3i({ c[0] + i, c[1] + j, c[2] + k }), OnLevel);
					auto m = this->surfacePoints.template findVoxel<OnLevel>(h);

					if (!m.isEnd()) {
						size_t gxp = m.template getLevelInfo<OnLevel>().pointCount;
						hits += gxp;
						if (count_max != 0 && hits >= count_max) {
							return hits;
						}
					}
				}
			}
		}

		return hits;
	}

	/*
	Remove all the voxels in the vector.

	OnLevel: On which level are the voxel to erase?
	*/
	template<int OnLevel>
	size_t removeVoxelsInList(const std::vector<cv::Vec3i> &toRemove) {
		size_t removed = 0;

		for (const auto &k : toRemove) {
			cv::Vec3f p = this->retrievePoint(k, OnLevel);
			removed += this->surfacePoints.template eraseVoxel<OnLevel>(p);
			std::cout << removed << std::endl;
		}
		return removed;
	}

	/*
	Filter outliers using countNeighborsFor. Essentially delete points that have to few close points.

	OnLevel: On which level to invoke countNeighborsFor
	l1radius: The radius passed on to countNeighborsFor
	minhits: The minimum number of close points a voxel needs to survive
	*/
	template<int OnLevel>
	std::size_t filterOutliers(const int l1radius, const int minhits) {
		// This thing is dependent on order, cause as outliers are removed, other points
		// that were before not outliers might become outliers. Anyway this is not handled here
		// for the sake of easy code and speed.

		std::vector<cv::Vec3i> toRemove;
		auto it = this->surfacePoints.treeIteratorBegin();
		while (!it.isEnd()) {
			cv::Vec3i c = it.template getLevelInfo<OnLevel>().voxelPosition;
			size_t hits = countNeighborsFor<OnLevel>(c, l1radius, minhits);

			if (hits < minhits) {
				toRemove.push_back(c);
			}

			it.template jump<OnLevel>();
		}

		return removeVoxelsInList<OnLevel>(toRemove);
	}

	/*
	Same as filterOutliers above, but check specifies a list of points for which an outlier check should be performed.
	*/
	template<int OnLevel>
	std::size_t filterOutliers(const int l1radius, const int minhits, const std::vector<ScenePoint>& check) {
		std::vector<cv::Vec3i> toRemove;

		std::unordered_set<cv::Vec3i, VecHash> toCheck;
		for (const ScenePoint &p : check) {
			toCheck.insert(this->retrieveVoxel(p.position, OnLevel));
		}

		for (const cv::Vec3i &c : toCheck) {
			cv::Vec3f h = this->retrievePoint(c, OnLevel);
			auto m = this->surfacePoints.template findVoxel<OnLevel>(h);

			if (!m.isEnd()) {
				size_t hits = countNeighborsFor<OnLevel>(c, l1radius, minhits);

				if (hits < minhits) {
					toRemove.push_back(c);
				}
			}
		}

		return removeVoxelsInList<OnLevel>(toRemove);
	}

	/*
	Export the pointcloud as an xyz file.
	*/
	void export_xyz(std::string path) {
		std::ofstream f(path, std::ios_base::out);
		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			cv::Vec3f u = it->first;
			cv::Vec3f v = it->second.position;
			cv::Vec3f n = it->second.normal;
			n = n / (cv::norm(n) + 0.0001);
			n *= it->second.numhits;
			f << v[0] << " " << v[1] << " " << v[2] << " " << n[0] << " " << n[1] << " " << n[2] << std::endl;
		}
		f.close();
	}
	
	/*
	Import an xyz file into the scene.
	*/
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
				//this->addPoint(v, n, 1.0);
				short numhits = std::round(cv::norm(n));
				n /= cv::norm(n);
				ScenePoint q(v, n, numhits);
				this->addPoint(q);
			}

			idx++;
		}
		if (idx % 6 != 0) {
			throw "Something was wrong when reading the file in import_xyz";
		}
		f.close();
	}

	void normalizeNormals() {
		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			it->second.normal = it->second.normal / (cv::norm(it->second.normal) + 0.00001);
		}
	}

	/*
	Add a single scenepoint to the pointcloud. If the voxel to which the point is 
	added is already in use, the mean of the positions and normals of all added points is computed
	and numhits is increased by the number of hits of the added ScenePoint. (probably 1)
	*/
	inline void addPoint(const ScenePoint& q) {
		auto it = this->surfacePoints.template findVoxel<Levels>(q.position);

		if (it.isEnd()) {
			this->surfacePoints.insert_or_update(q.position, q);
		}
		else {
			const ScenePoint& old = it->second;
			short newhits = old.numhits + q.numhits;
			cv::Vec3f p = (q.numhits * q.position + old.position * old.numhits) / (newhits);
			cv::Vec3f n = (q.numhits * q.normal + old.normal * old.normal) / (newhits);
			this->surfacePoints.insert_or_update(q.position, ScenePoint(p, n, newhits));
		}
	}

	/*
	Add all points in newPoints to the Pointcloud, but only add once to each voxel, even if
	points in the same voxel are contained multiple times in newPoints.
	*/
	inline void addAllSingleCount(const std::vector<ScenePoint>& newPoints) {
        std::unordered_set<cv::Vec3i, VecHash> addedPointsBuffer;

		for (const ScenePoint& p : newPoints) {
			cv::Vec3i u = this->retrieveVoxel(p.position, Levels);
			if (addedPointsBuffer.find(u) == addedPointsBuffer.end()) {
				addedPointsBuffer.insert(u);
				this->addPoint(p);
			}
		}
	}

	void refineNormals(float radius, int knearest) {
		constexpr int OnLevel = 0;

		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			cv::Vec3f current = it->second.position;
			auto close = this->findNeighborsFor<OnLevel>(current, radius);
			std::priority_queue<NormalPrio, std::vector<NormalPrio>> prioqueue;

			for (auto it2 = close.begin(); it2 != close.end(); it2++) {
				auto it3 = *it2;
				int pointcount = it3.template getLevelInfo<OnLevel>().pointCount;
				for (int i = 0; i < pointcount; i++, it3++) {
					NormalPrio g;
					g.pos = it3->second.position;
					g.dist = cv::norm(g.pos - current, cv::NORM_L2);
					prioqueue.push(g);
				}
			}

			if (prioqueue.size() < knearest) {
				it->second.normal = cv::Vec3f::zeros();
				it->second.numhits = 0;
			}
			else {
				cv::Vec3d mean = cv::Vec3d::zeros();
				cv::Mat A(3, knearest, CV_32F);
				for (int i = 0; i < knearest; i++) {
					NormalPrio np = prioqueue.top();
					prioqueue.pop();

					cv::Vec3f p = np.pos;
					A.at<float>(0, i) = p[0];
					A.at<float>(1, i) = p[1];
					A.at<float>(2, i) = p[2];
					mean += p;
				}
				mean = mean * (1.0 / knearest);
				for (int i = 0; i < knearest; i++) {
					A.at<float>(0, i) -= mean[0];
					A.at<float>(1, i) -= mean[1];
					A.at<float>(2, i) -= mean[2];
				}

				cv::SVD svd(A);

				cv::Vec3f n = svd.u.col(2);
				n = n / cv::norm(n);

				float d = n.dot(it->second.normal);
				if (d < 0) {
					n = -n;
				}
				it->second.normal = n;
			}
		}
	}
	
	/*
	Erase all points that have less than minView hits.
	*/
	std::size_t filterNumviews(const short minView) {
		auto it = this->surfacePoints.treeIteratorBegin();
		std::vector<cv::Vec3i> toRemove;

		while (!it.isEnd()) {
			if (it->second.numhits < minView) {
				toRemove.push_back(it->first);
			}
			it++;
		}

		return removeVoxelsInList<Levels>(toRemove);
	}

	/*
	Erase all points that have less than minView hits if they are also in check. (check is allowed to
	contain voxels that are actually not part of the pointcloud)
	*/
	std::size_t filterNumviews(const short minView, const std::vector<ScenePoint>& check) {
		std::vector<cv::Vec3i> toRemove;

		std::unordered_set<cv::Vec3i, VecHash> toCheck;
		for (const ScenePoint &p : check) {
			toCheck.insert(this->retrieveVoxel(p.position, Levels));
		}

		for (const cv::Vec3i &c : toCheck) {
			cv::Vec3f h = this->retrievePoint(c, Levels);
			auto m = this->surfacePoints.template findVoxel<Levels>(h);

			if (!m.isEnd()) {
				if (m->second.numhits < minView) {
					toRemove.push_back(c);
				}
			}
		}

		return removeVoxelsInList<Levels>(toRemove);
	}
};
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include "helpers.h"
#include "HierarchicalVoxelGrid.h"


template<int Levels, typename NodeStorage>
class Scene : public HierachicalVoxelGrid<Levels, NodeStorage, ScenePoint> {
public:
	Scene(double voxelSideLength, std::vector<int> indexBlocks) : HierachicalVoxelGrid<Levels, NodeStorage, ScenePoint>(voxelSideLength, indexBlocks) {

	}
	
	std::size_t filterConfidence() {
		std::vector<cv::Vec3i> toRemove;
		float avgConfidence = 0;

		size_t psize = this->surfacePoints.getPointCount();
		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			avgConfidence += it->second.confidence * (1.0 / psize);
		}

		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			if (it->second.confidence <= avgConfidence * 0.7) {
				toRemove.push_back(it->first);
			}
		}

		for (auto g : toRemove) {
			this->surfacePoints.erase(this->retrievePoint(g, Levels));
		}

		return toRemove.size();
	}

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

	template<int OnLevel>
	size_t removeVoxelsInList(const std::vector<cv::Vec3i> &toRemove) {
		size_t removed = 0;

		for (const auto &k : toRemove) {
			cv::Vec3f p = this->retrievePoint(k, OnLevel);
			removed += this->surfacePoints.template eraseVoxel<OnLevel>(p);
		}
		return removed;
	}

	
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

	void export_xyz(std::string path) {
		std::ofstream f(path, std::ios_base::out);
		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			cv::Vec3f u = it->first;
			cv::Vec3f v = it->second.position;
			cv::Vec3f n = it->second.normal;
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
				//this->addPoint(v, n, 1.0);
				ScenePoint q(v, n, 1.0);
				this->addPoint(q);
			}

			idx++;
		}
		if (idx % 6 != 0) {
			throw "Something was wrong when reading the file in import_xyz";
		}
		f.close();
	}

	inline void addPoint(const ScenePoint& q) {
		auto current = this->surfacePoints.findVoxel<Levels>(q.position);
		if (current.isEnd()) {
			this->surfacePoints.insert_or_update(q.position, q);
		}
		else {
			float oldconfidence = current->second.confidence;
			float newconfidence = oldconfidence + q.confidence;
			current->second.confidence = newconfidence;
			current->second.normal = (oldconfidence * current->second.normal + q.confidence * q.normal) / newconfidence;
			current->second.position = (oldconfidence * current->second.position + q.confidence * q.position) / newconfidence;
		}
	}
};
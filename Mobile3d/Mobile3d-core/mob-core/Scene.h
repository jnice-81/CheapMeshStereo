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
	float confidence = 0;
};

class TreeIterator;

template<int Levels>
class HierachicalVoxelGrid {
protected:

	template<int CurrentLevel, int MaxLevel>
	class TreeIterator;

	template<int CurrentLevel, int MaxLevel>
	class TreeLevel : public std::unordered_map<cv::Vec3i, TreeLevel<CurrentLevel + 1, MaxLevel>, VecHash> {
	private:
		HierachicalVoxelGrid *parent;
		size_t num_points = 0;
	public:
		TreeLevel(HierachicalVoxelGrid *parent)
		{
			this->parent = parent;
		}

		inline size_t getPointCount() const {
			return num_points;
		}

		bool insert_or_update(cv::Vec3f p, const cv::Vec3f normal, const float confidence) {
			cv::Vec3i pI = parent->retrieveVoxel(p, CurrentLevel);
			auto c = this->find(pI);
			if (c != this->end()) {
				if (c->second.insert_or_update(p, normal, confidence)) {
					num_points++;
					return true;
				}
				else {
					return false;
				}
			}
			else {
				auto n = TreeLevel<CurrentLevel + 1, MaxLevel>(parent);
				n.insert_or_update(p, normal, confidence);
				this->insert(std::make_pair(pI, n));
				num_points++;
				return true;
			}
		}

		TreeIterator<CurrentLevel, MaxLevel> treeIteratorBegin() {
			return TreeIterator<CurrentLevel, MaxLevel>(this->begin(), this->end());
		}

	};

	template<int MaxLevel>
	class TreeLevel<MaxLevel, MaxLevel> : public std::unordered_map<cv::Vec3i, ScenePoint, VecHash> {
	private:
		HierachicalVoxelGrid *parent;
	public:
		TreeLevel(HierachicalVoxelGrid *parent)
		{
			this->parent = parent;
		}

		inline size_t getPointCount() const {
			return this->size();
		}

		bool insert_or_update(cv::Vec3f p, const cv::Vec3f normal, const float confidence) {
			cv::Vec3i pI = parent->retrieveVoxel(p, MaxLevel);
			auto it = this->find(pI);
			if (it != this->end()) {
				float oldconfidence = it->second.confidence;
				it->second.confidence = oldconfidence + confidence;
				it->second.normal = (oldconfidence * it->second.normal + confidence * normal) / it->second.confidence;
				return false;
			}
			else {
				ScenePoint q;
				q.normal = normal;
				q.confidence = confidence;
				this->insert(std::make_pair(pI, q));
				return true;
			}
		}

		TreeIterator<MaxLevel, MaxLevel> treeIteratorBegin() {
			return TreeIterator<MaxLevel, MaxLevel>(this->begin(), this->end());
		}
	};

	template<int CurrentLevel, int MaxLevel>
	class TreeIterator {
	private:
		TreeIterator<CurrentLevel + 1, MaxLevel> lowerIt;
		typename TreeLevel<CurrentLevel, MaxLevel>::iterator upperIt;
		typename TreeLevel<CurrentLevel, MaxLevel>::iterator end;
	public:
		TreeIterator(typename TreeLevel<CurrentLevel, MaxLevel>::iterator start, typename TreeLevel<CurrentLevel, MaxLevel>::iterator end) {
			this->upperIt = start;
			this->end = end;
			if (upperIt != end) {
				this->lowerIt = start->second.treeIteratorBegin();
			}
		}

		TreeIterator() {}

		void operator++(int) {
			lowerIt++;
			if (lowerIt.isEnd()) {
				upperIt++;
				if (upperIt != end) {
					lowerIt = upperIt->second.treeIteratorBegin();
				}
			}
		}

		std::pair<const cv::Vec3i, ScenePoint>& operator*() {
			return *lowerIt;
		}

		std::pair<const cv::Vec3i, ScenePoint>* operator->() {
			return &*lowerIt;
		}

		bool isEnd() {
			return this->upperIt == this->end;
		}
	};

	template<int MaxLevel>
	class TreeIterator<MaxLevel, MaxLevel> {
	private:
		typename TreeLevel<MaxLevel, MaxLevel>::iterator it;
		typename TreeLevel<MaxLevel, MaxLevel>::iterator end;
	public:
		TreeIterator(typename TreeLevel<MaxLevel, MaxLevel>::iterator start, typename TreeLevel<MaxLevel, MaxLevel>::iterator end) {
			this->it = start;
			this->end = end;
		}

		TreeIterator() {}

		void operator++(int) {
			it++;
		}

		std::pair<const cv::Vec3i, ScenePoint>* operator->() {
			return &*it;
		}

		std::pair<const cv::Vec3i, ScenePoint>& operator*() {
			return *it;
		}

		bool isEnd() {
			return this->it == end;
		}
	};


	std::vector<double> preprocVoxelSizes;
	TreeLevel<0, Levels> surfacePoints;
	double voxelSideLength;

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

public:
	HierachicalVoxelGrid(double voxelSideLength, std::vector<int> &indexBlocks) : surfacePoints(this) {
		preprocVoxelSizes.resize(1 + indexBlocks.size());
		preprocVoxelSizes[indexBlocks.size() - 1 + 1] = voxelSideLength;
		this->voxelSideLength = voxelSideLength;
		for (int j = (int)preprocVoxelSizes.size() - 2; j >= 0; j--) {
			preprocVoxelSizes[j] = preprocVoxelSizes[j + 1] * indexBlocks[j];
		}
	}

	// Level: Starts at 0 goes downward
	inline cv::Vec3i retrieveVoxel(const cv::Vec3f p, const int level) const {
		return floatToIntVec<int, float, 3>(p / preprocVoxelSizes[level]);
	}

	inline void addPoint(const cv::Vec3f point, const cv::Vec3f normal, const float confidence) {
		surfacePoints.insert_or_update(point, normal, confidence);
	}

	inline TreeLevel<0, Levels>& getSceneData() {
		return surfacePoints;
	}
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

template<int Levels>
class Scene : public HierachicalVoxelGrid<Levels> {
public:
	Scene(double voxelSideLength, std::vector<int> indexBlocks) : HierachicalVoxelGrid<Levels>(voxelSideLength, indexBlocks) {
		
	}

	inline cv::Vec3f addVoxelCenter(const cv::Vec3f voxel) const {
		const float center = (float)this->voxelSideLength * 0.5f;
		const cv::Vec3f toCenter(center, center, center);
		return voxel + toCenter;
	}

	inline cv::Vec3f voxelToPoint(const cv::Vec3i voxelIdx) const {
		cv::Vec3f p = voxelIdx;
		return addVoxelCenter(p * this->voxelSideLength);
	}

	inline cv::Vec3f getCenterOfVoxel(const cv::Vec3f point) const {
		cv::Vec3i q = this->template floatToIntVec<int, float, 3>(point / this->voxelSideLength);
		return addVoxelCenter((cv::Vec3f)q * this->voxelSideLength);
	}

	/*
	std::size_t filterConfidence(const float minConfidence) {
		std::vector<cv::Vec3i> toRemove;

		for (auto it = surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			if (it->second.confidence <= minConfidence) {
				toRemove.push_back(it->first);
			}
		}

		for (auto g : toRemove) {
			surfacePoints.treeErase(g);
		}

		return toRemove.size();
	}

	std::size_t filterOutliers(const int l1radius, const int minhits) {
		// This thing is dependent on order, cause as outliers are removed, other points
		// that were before not outliers might become outliers. Anyway this is not handled here
		// for the sake of easy code and speed.
		std::vector<cv::Vec3i> toRemove;

		for (auto it = surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			cv::Vec3i c = it->first;
			int hits = 0;

			for (int i = -l1radius; i <= l1radius; i++) {
				for (int j = -l1radius; j <= l1radius; j++) {
					for (int k = -l1radius; k <= l1radius; k++) {
						cv::Vec3i h({ c[0] + i, c[1] + j, c[2] + k });
						if (surfacePoints.treeFind(h)) {
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
			surfacePoints.treeErase(*it);
		}

		return toRemove.size();
	}
	
	*/

	void export_xyz(std::string path) {
		std::ofstream f(path, std::ios_base::out);
		for (auto it = this->surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
			cv::Vec3f u = it->first;
			cv::Vec3f v = addVoxelCenter(u * this->voxelSideLength);
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
				this->addPoint(v, n, 1.0);
			}

			idx++;
		}
		if (idx % 6 != 0) {
			throw "Something was wrong when reading the file in import_xyz";
		}
		f.close();
	}

	inline double getVoxelSideLength() const {
		return this->voxelSideLength;
	}

	/*
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

		for (auto it = surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
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
	*/
};
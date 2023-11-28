#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
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
	ScenePoint(cv::Vec3f position, cv::Vec3f normal, float confidence = 0)
	{
		this->position = position;
		this->normal = normal;
		this->confidence = confidence;
	}

	ScenePoint() {

	}

	cv::Vec3f position;
	cv::Vec3f normal;
	float confidence = 0;
};

class TreeIterator;

template<int Levels, typename NodePayload>
class HierachicalVoxelGrid {
public:
	class LevelInfo {
	public:
		LevelInfo(cv::Vec3i voxelPosition, size_t pointCount, size_t childCount) {
			this->voxelPosition = voxelPosition;
			this->pointCount = pointCount;
			this->childCount = childCount;
		}

		cv::Vec3i voxelPosition;
		size_t pointCount;
		size_t childCount;
	};

protected:
	template<int CurrentLevel, int MaxLevel>
	class TreeIterator;

	template<int CurrentLevel, int MaxLevel>
	class TreeLevel : public std::unordered_map<cv::Vec3i, TreeLevel<CurrentLevel + 1, MaxLevel>, VecHash> {
	private:
		HierachicalVoxelGrid *parent;
		size_t num_points = 0;
		NodePayload payload;
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

		template<int SelectLevel>
		size_t eraseVoxel(cv::Vec3f p) {
			cv::Vec3i pI = parent->retrieveVoxel(p, CurrentLevel);
			auto c = this->find(pI);
			if constexpr (SelectLevel == CurrentLevel) {
				size_t points;
				points = c->second.getPointCount();
				this->erase(pI);
				num_points -= points;
				return points;
			}
			else {
				size_t erased = c->second.template eraseVoxel<SelectLevel>(p);
				if (c->second.getPointCount() == 0) {
					this->erase(pI);
				}
				num_points -= erased;
				return erased;
			}
		}

		template<int SelectLevel>
		TreeIterator<SelectLevel, MaxLevel> findSuperVoxel(const cv::Vec3f p) {
			cv::Vec3i pI = parent->retrieveVoxel(p, CurrentLevel);
			auto g = this->find(pI);
			if constexpr (SelectLevel == CurrentLevel) {
				if (g != this->end()) {
					return TreeIterator<SelectLevel, MaxLevel>(g, this->end());
				}
				else {
					return TreeIterator<SelectLevel, MaxLevel>();
				}
			}
			else {
				if (g != this->end()) {
					return g->second.template findSuperVoxel<SelectLevel>(p);
				}
				else {
					return TreeIterator<SelectLevel, MaxLevel>();
				}
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

		template <int SelectLevel>
		size_t eraseVoxel(cv::Vec3f p) {
			cv::Vec3i pI = parent->retrieveVoxel(p, MaxLevel);
			this->erase(pI);
			return 1;
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
		bool empty = false;

		inline void increaseUpperIt() {
			upperIt++;
			if (upperIt != end) {
				lowerIt = upperIt->second.treeIteratorBegin();
			}
		}
	public:
		TreeIterator() {
			empty = true;
		}

		TreeIterator(typename TreeLevel<CurrentLevel, MaxLevel>::iterator start, typename TreeLevel<CurrentLevel, MaxLevel>::iterator end) {
			this->upperIt = start;
			this->end = end;
			if (upperIt != end) {
				this->lowerIt = start->second.treeIteratorBegin();
			}
		}

		void operator++(int) {
			lowerIt++;
			if (lowerIt.isEnd()) {
				increaseUpperIt();
			}
		}

		template<int SelectLevel>
		bool jump() {
			if constexpr (SelectLevel == CurrentLevel) {
				increaseUpperIt();
				return upperIt != end;
			}
			else {
				if (!lowerIt.template jump<SelectLevel>()) {
					increaseUpperIt();
					return upperIt != end;
				}
				return true;
			}
		}

		template<int SelectLevel>
		LevelInfo getLevelInfo() {
			if constexpr (SelectLevel == CurrentLevel) {
				return LevelInfo(upperIt->first, upperIt->second.getPointCount(), upperIt->second.size());
			}
			else {
				return lowerIt.template getLevelInfo<SelectLevel>();
			}
		}

		std::pair<const cv::Vec3i, ScenePoint>& operator*() {
			return *lowerIt;
		}

		std::pair<const cv::Vec3i, ScenePoint>* operator->() {
			return &*lowerIt;
		}

		bool isEnd() {
			return empty || this->upperIt == this->end;
		}
	};

	template<int MaxLevel>
	class TreeIterator<MaxLevel, MaxLevel> {
	private:
		typename TreeLevel<MaxLevel, MaxLevel>::iterator it;
		typename TreeLevel<MaxLevel, MaxLevel>::iterator end;
		bool empty = false;
	public:
		TreeIterator() {
			empty = true;
		}

		TreeIterator(typename TreeLevel<MaxLevel, MaxLevel>::iterator start, typename TreeLevel<MaxLevel, MaxLevel>::iterator end) {
			this->it = start;
			this->end = end;
		}

		void operator++(int) {
			it++;
		}

		template<int SelectLevel>
		bool jump() {
			throw "Likely not what you wanted to do; jump hit maxlevel";
		}

		template<int SelectLevel>
		LevelInfo getLevelInfo() {
			throw "Likely not what you wanted to do; getLevel hit maxlevel";
		}

		std::pair<const cv::Vec3i, ScenePoint>* operator->() {
			return &*it;
		}

		std::pair<const cv::Vec3i, ScenePoint>& operator*() {
			return *it;
		}

		bool isEnd() {
			return empty || this->it == end;
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
		cv::Vec3i r = floatToIntVec<int, float, 3>(p / preprocVoxelSizes[level]);
		return r;
	}

	inline cv::Vec3f retrieveVoxelCenter(const int level) const {
		const float center = (float)this->preprocVoxelSizes[level] * 0.5f;
		const cv::Vec3f toCenter(center, center, center);
		return toCenter;
	}

	inline cv::Vec3f retrievePoint(const cv::Vec3i c, const int level) const {
		cv::Vec3f r = ((cv::Vec3f)c) * preprocVoxelSizes[level] + retrieveVoxelCenter(level);
		return r;
	}

	inline void addPoint(const cv::Vec3f point, const cv::Vec3f normal, const float confidence) {
		surfacePoints.insert_or_update(point, normal, confidence);
	}

	inline void treeErase(const cv::Vec3f point) {
		surfacePoints.erase(point);
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

template<int Levels, typename NodeStorage>
class Scene : public HierachicalVoxelGrid<Levels, NodeStorage> {
public:
	Scene(double voxelSideLength, std::vector<int> indexBlocks) : HierachicalVoxelGrid<Levels, NodeStorage>(voxelSideLength, indexBlocks) {
		
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
			this->treeErase(voxelToPoint(g));
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
					auto m = this->surfacePoints.template findSuperVoxel<OnLevel>(h);

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

		for (const auto k : toRemove) {
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
		for (const ScenePoint p : check) {
			toCheck.insert(this->retrieveVoxel(p.position, OnLevel));
		}

		for (const cv::Vec3i c : toCheck) {
			cv::Vec3f h = this->retrievePoint(c, OnLevel);
			auto m = this->surfacePoints.template findSuperVoxel<OnLevel>(h);

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
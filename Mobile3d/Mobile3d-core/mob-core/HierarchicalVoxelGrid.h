#pragma once

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

class ScenePoint {
public:
	ScenePoint(cv::Vec3f position, cv::Vec3f normal, short numhits)
	{
		this->position = position;
		this->normal = normal;
		this->numhits = numhits;
	}

	ScenePoint() {

	}

	cv::Vec3f position;
	cv::Vec3f normal;
	short numhits;
};

class TreeIterator;

template<int Levels, typename NodePayload, typename VoxelPayload>
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

	template<int CurrentLevel, int MaxLevel>
	class TreeIterator;

	template<int CurrentLevel, int MaxLevel>
	class TreeLevel : public std::unordered_map<cv::Vec3i, TreeLevel<CurrentLevel + 1, MaxLevel>, VecHash> {
	private:
		HierachicalVoxelGrid* parent;
		size_t num_points = 0;
		NodePayload payload;
	public:
		TreeLevel(HierachicalVoxelGrid* parent)
		{
			this->parent = parent;
		}

		inline size_t getPointCount() const {
			return num_points;
		}

		virtual void clear() {
			std::unordered_map<cv::Vec3i, TreeLevel<CurrentLevel + 1, MaxLevel>, VecHash>::clear();
			this->num_points = 0;
		}

		bool insert_or_update(cv::Vec3f p, VoxelPayload v) {
			cv::Vec3i pI = parent->retrieveVoxel(p, CurrentLevel);
			auto c = this->find(pI);
			if (c != this->end()) {
				if (c->second.insert_or_update(p, v)) {
					num_points++;
					return true;
				}
				else {
					return false;
				}
			}
			else {
				auto n = TreeLevel<CurrentLevel + 1, MaxLevel>(parent);
				n.insert_or_update(p, v);
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
		TreeIterator<SelectLevel, MaxLevel> findVoxel(const cv::Vec3f p) {
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
					return g->second.template findVoxel<SelectLevel>(p);
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
	class TreeLevel<MaxLevel, MaxLevel> : public std::unordered_map<cv::Vec3i, VoxelPayload, VecHash> {
	private:
		HierachicalVoxelGrid* parent;
	public:
		TreeLevel(HierachicalVoxelGrid* parent)
		{
			this->parent = parent;
		}

		inline size_t getPointCount() const {
			return this->size();
		}

		template<int SelectLevel>
		TreeIterator<MaxLevel, MaxLevel> findVoxel(const cv::Vec3f p) {
			cv::Vec3i pI = parent->retrieveVoxel(p, MaxLevel);
			auto g = this->find(pI);
			if (g != this->end()) {
				return TreeIterator<MaxLevel, MaxLevel>(g, this->end());
			}
			else {
				return TreeIterator<MaxLevel, MaxLevel>();
			}
		}

		bool insert_or_update(cv::Vec3f p, VoxelPayload v) {
			cv::Vec3i pI = parent->retrieveVoxel(p, MaxLevel);
			auto it = this->find(pI);
			if (it != this->end()) {
				it->second = v;
				return false;
			}
			else {
				this->insert(std::make_pair(pI, v));
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

		std::pair<const cv::Vec3i, VoxelPayload>& operator*() {
			return *lowerIt;
		}

		std::pair<const cv::Vec3i, VoxelPayload>* operator->() {
			return &*lowerIt;
		}

		bool isEnd() const {
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

		std::pair<const cv::Vec3i, VoxelPayload>* operator->() {
			return &*it;
		}

		std::pair<const cv::Vec3i, VoxelPayload>& operator*() {
			return *it;
		}

		bool isEnd() {
			return empty || this->it == end;
		}
	};

	HierachicalVoxelGrid(double voxelSideLength, std::vector<int> indexBlocks) : surfacePoints(this) {
		preprocVoxelSizes.resize(1 + indexBlocks.size());
		preprocVoxelSizes[indexBlocks.size() - 1 + 1] = voxelSideLength;
		for (int j = (int)preprocVoxelSizes.size() - 2; j >= 0; j--) {
			preprocVoxelSizes[j] = preprocVoxelSizes[j + 1] * indexBlocks[j];
		}
	}

	// Level: Starts at 0 goes downward
	inline cv::Vec3i retrieveVoxel(const cv::Vec3f &p, const int level) const {
		cv::Vec3i r = floatToIntVec<int, float, 3>(p / preprocVoxelSizes[level]);
		return r;
	}

	inline cv::Vec3f retrieveVoxelCenter(const int level) const {
		const float center = (float)this->preprocVoxelSizes[level] * 0.5f;
		const cv::Vec3f toCenter(center, center, center);
		return toCenter;
	}

	inline cv::Vec3f retrievePoint(const cv::Vec3i &c, const int level) const {
		cv::Vec3f r = ((cv::Vec3f)c) * preprocVoxelSizes[level] + retrieveVoxelCenter(level);
		return r;
	}

	inline cv::Vec3f retrieveCornerPoint(const cv::Vec3i &c, const int level) const {
		cv::Vec3f r = ((cv::Vec3f)c) * preprocVoxelSizes[level];
		return r;
	}

	inline double retrieveVoxelSidelength(const int level) const {
		return this->preprocVoxelSizes[level];
	}

	TreeLevel<0, Levels> surfacePoints;

	private:
		std::vector<double> preprocVoxelSizes;
};
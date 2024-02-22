#pragma once

/*
A class implementing a simple hash function for 3d integer vectors
*/
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

/*
Converts a floating point vector to an integer vector. (opencv seems to be buggy there?)

It: The integer type
Ft: The floating point type
Dim: The dimensionality of the vector
*/
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

class TreeIterator;

/*
A storage class implementing a sparsly populated voxel grid with multiple levels.
Annoying to explain in a comment, see the report.

Levels: The number of Levels the Grid has
VoxelPayload: Data to be stored on the lowest level voxels
*/
template<int Levels, typename VoxelPayload>
class HierachicalVoxelGrid {
public:
	/*
	Storage class holding information about a particular super-voxel.

	voxelPosition: The index of the voxel (loosly related to its position in 3d)
	pointCount: The number of (lowest level) points this voxel contains.
	childCount: The number of children this voxel contains. If this is the second lowest level this 
		equals pointCount. Otherwise it is how many of the voxels in the next lower level contain at least one
		point.
	*/
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

	/*
	The actual storage class used by HierarchicalVoxelGrid.
	*/
	template<int CurrentLevel, int MaxLevel>
	class TreeLevel : private std::unordered_map<cv::Vec3i, TreeLevel<CurrentLevel + 1, MaxLevel>, VecHash> {
	private:
		HierachicalVoxelGrid* parent;
		size_t num_points = 0;
	public:
		template<int CurrentLevelO, int MaxLevelO>
		friend class TreeIterator;

		TreeLevel(HierachicalVoxelGrid* parent)
		{
			this->parent = parent;
		}

		/*
		Get the number of lowest level voxels this voxel contains
		*/
		inline size_t getPointCount() const {
			return num_points;
		}

		/*
		Erase all contents.
		*/
		virtual void clear() {
			std::unordered_map<cv::Vec3i, TreeLevel<CurrentLevel + 1, MaxLevel>, VecHash>::clear();
			this->num_points = 0;
		}

		/*
		Add data to the lowest level voxel which contains p. Either creates
		or overwrites the data.

		p: A vector indicating a position
		v: The data to store
		*/
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

		/*
		Erase the voxel that contains point p.

		SelectLevel: On which level to erase a voxel
		p: A position which the voxel contains
		*/
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

		/*
		Get an iterator to the voxel that contains p. This is simply a default iterator to the first
		lowest level voxel the selected voxel contains. If you want to ensure to only read lowest level voxels
		that are contained in the selected voxel you will hence have to use getLevelInfo on the returned iterator
		and increase the iterator only by the number of lowest level voxels the selected voxel contains. 

		SelectLevel: The level on which the voxel should be selected
		p: A position contained by the voxel which should be selected
		*/
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

		/*
		Get an iterator to the beginning of the current level.
		*/
		TreeIterator<CurrentLevel, MaxLevel> treeIteratorBegin() {
			return TreeIterator<CurrentLevel, MaxLevel>(this->begin(), this->end());
		}

	};

	/*
	Comments are same as above. This implements the handling of the lowest level voxels.
	*/
	template<int MaxLevel>
	class TreeLevel<MaxLevel, MaxLevel> : private std::unordered_map<cv::Vec3i, VoxelPayload, VecHash> {
	private:
		HierachicalVoxelGrid* parent;
	public:
		template<int CurrentLevelO, int MaxLevelO>
		friend class TreeIterator;

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

	/*
	An iterator type for the TreeLevel data structure. Does not strictly conform to the c++ iterator
	standard. (Mostly cause slow). An iterator of this type always points to a lowest level voxel (or end),
	but because a lowest level voxel is contained by higher level voxels, the iterator implicitly also points to those.
	(And can be used to manipulate them)
	*/
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

		/*
		Advance the iterator to the next element.
		*/
		void operator++(int) {
			lowerIt++;
			if (lowerIt.isEnd()) {
				increaseUpperIt();
			}
		}

		/*
		Advance the iterator to the first element of the next higher level voxel.

		SelectLevel: The Level on which to advance the iterator
		*/
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

		/*
		Returns information about the voxel this iterator points to. 
		Note that this is only implemented for levels higher than the lowest level.

		SelectLevel: The level for which to get information
		*/
		template<int SelectLevel>
		LevelInfo getLevelInfo() {
			if constexpr (SelectLevel == CurrentLevel) {
				return LevelInfo(upperIt->first, upperIt->second.getPointCount(), upperIt->second.size());
			}
			else {
				return lowerIt.template getLevelInfo<SelectLevel>();
			}
		}

		/*
		Get the data stored in the lowest level voxel
		*/
		std::pair<const cv::Vec3i, VoxelPayload>& operator*() {
			return *lowerIt;
		}

		/*
		Get the data stored in the lowest level voxel
		*/
		std::pair<const cv::Vec3i, VoxelPayload>* operator->() {
			return &*lowerIt;
		}

		/*
		Check if the iterator points to end. (Is not defined)
		*/
		bool isEnd() const {
			return empty || this->upperIt == this->end;
		}
	};

	/*
	Same comments as above. Implements the iterator for the lowest level. 
	jump will not work on this level (same operation as ++);
	getLevelInfo will also not work (returned information mostly trivial, so there is not really a use case for this)
	*/
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

	/*
	Initialize a Hierachical Voxel Grid.

	voxelSideLength: A double indicating the sidelength of the lowest level voxels
	indexBlocks: A vector of int of the same size as Levels. Indicates for each level how many subvoxels
		(in 1d) a voxel contains. Suppose for example Levels = 2 and indexBlocks = {10, 5}.
		Then the first level would contain voxels with a sidelength of 10 * 5 * voxelsidelength, the
		second level with 5 * voxelsidelength, and lowest level voxels have a sidelength of voxelsidelength.
	*/
	HierachicalVoxelGrid(double voxelSideLength, std::vector<int> indexBlocks) : surfacePoints(this) {
		this->init(voxelSideLength, indexBlocks);
	}

	/*
	Erase all data and change voxel sidelengths. (Same as constructor)
	*/
	void reset(double voxelSideLength, std::vector<int> indexBlocks) {
		this->surfacePoints.clear();
		this->init(voxelSideLength, indexBlocks);
	}

	/*
	Get the index of the voxel which contains p on a given level. Level starts at 0 
	(i.e. the largest voxels are on level 0, smallest on level Levels)
	*/
	inline cv::Vec3i retrieveVoxel(const cv::Vec3f &p, const int level) const {
		cv::Vec3i r = floatToIntVec<int, float, 3>(p / preprocVoxelSizes[level]);
		return r;
	}

	/*
	Get the vector pointing from the corner of a voxel to its center for a given level.
	Essentially for internal use, seems unlikely to be used otherwise.
	*/
	inline cv::Vec3f retrieveVoxelCenter(const int level) const {
		const float center = (float)this->preprocVoxelSizes[level] * 0.5f;
		const cv::Vec3f toCenter(center, center, center);
		return toCenter;
	}

	/*
	Retrieve the point at the center of a given voxel.

	c: The index of the voxel
	level: The level of the voxel
	*/
	inline cv::Vec3f retrievePoint(const cv::Vec3i &c, const int level) const {
		cv::Vec3f r = ((cv::Vec3f)c) * preprocVoxelSizes[level] + retrieveVoxelCenter(level);
		return r;
	}

	/*
	Retrieve the corner point of a voxel.

	c: The index of the voxel
	level: The level of the voxel
	*/
	inline cv::Vec3f retrieveCornerPoint(const cv::Vec3i &c, const int level) const {
		cv::Vec3f r = ((cv::Vec3f)c) * preprocVoxelSizes[level];
		return r;
	}

	/*
	Get the (de facto) sidelength of voxels on a level.

	level: The level 
	*/
	inline double retrieveVoxelSidelength(const int level) const {
		return this->preprocVoxelSizes[level];
	}

	/*
	Find all points which are closer in l1 distance to p than l1radius. Essentially simply includes all 
	Voxels that can contain any points defined by this cube. It is guaranteed that all points closer than
	l1radius are listed, and that each point only appears once. However points which are farther away than l1 distance
	might also be listed.
	(Note: For very small voxel sizes this function might have problems with numberical stability)

	OnLevel: On which level of the hierarchical grid to list all voxels which have a point that is closer than l1radius.
		This does not change the functionality, but has a high effect on speed. (A good trick to get fast lookups is to choose 
		a level where l1radius roughly corresponds to the voxelsidelength)
	p: The point in the center of the region which you are interested in
	l1radius: The (L1) radius in which to obtain points
	out: A vector of iterators containing iterators to all found voxels. If you want to iterate all points you should iterate 
		all elements of the vector, check the number of points on the Level OnLevel (using GetLevelInfo) and increase the iterator this number
		of times. 
	*/
	template <int OnLevel>
	void findNeighborsFor(const cv::Vec3f p, const float l1radius, std::vector<TreeIterator<OnLevel, Levels>>& out) {
		cv::Vec3f l1radi;
		cv::Vec3f cornerQuad;

		auto tmp = cv::Vec3f(l1radius, l1radius, l1radius);
		cornerQuad = p - tmp;
		l1radi = 2 * tmp;
		float voxelSidelength = this->retrieveVoxelSidelength(OnLevel);
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
					select.insert(this->retrieveVoxel(q, OnLevel));
				}
			}
		}

		out.reserve(select.size());
		for (const auto& g : select) {
			auto it = this->surfacePoints.findVoxel<OnLevel>(this->retrievePoint(g, OnLevel));
			if (!it.isEnd()) {
				out.push_back(it);
			}
		}
	}

	/*
	The actual data storage variable.
	*/
	TreeLevel<0, Levels> surfacePoints;

	private:
		std::vector<double> preprocVoxelSizes;

		void init(double voxelSideLength, std::vector<int> indexBlocks) {
			assert(indexBlocks.size() == Levels);
			preprocVoxelSizes.resize(1 + indexBlocks.size());
			preprocVoxelSizes[indexBlocks.size() - 1 + 1] = voxelSideLength;
			for (int j = (int)preprocVoxelSizes.size() - 2; j >= 0; j--) {
				preprocVoxelSizes[j] = preprocVoxelSizes[j + 1] * indexBlocks[j];
			}
		}
};
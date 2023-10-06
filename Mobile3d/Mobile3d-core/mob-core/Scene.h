

class ScenePoint {
public:
	cv::Vec3f tmp;

private:
	unsigned short hit_count;
};

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

class Scene {
public:
	Scene(double voxelSideLength) {
		this->voxelSideLength = voxelSideLength;
	}

	inline void addPoint(cv::Vec3f p) {
		cv::Vec3i q = p / voxelSideLength;
		ScenePoint s;
		s.tmp = p;
		surfacePoints.insert(std::make_pair(q, s));
	}

	cv::Mat directRender(View& v) {
		cv::Size imgsize = v.image.size();
		cv::Mat result = cv::Mat::zeros(imgsize, CV_32F);
		cv::Matx44d extrinsics;
		cv::Matx33d intrinsics;
		v.extrinsics.convertTo(extrinsics, CV_64F);
		v.intrinsics.convertTo(intrinsics, CV_64F);

		for (auto it = surfacePoints.begin(); it != surfacePoints.end(); it++) {
			cv::Vec3f p = it->second.tmp;
			cv::Vec4d hp = cv::Vec4d(p[0], p[1], p[2], 1.0);
			hp = extrinsics * hp;
			cv::Vec3d tmp = cv::Vec3d(hp.val) / hp[3];
			tmp = intrinsics * tmp;
			cv::Vec2i pdash = cv::Vec2d(tmp.val) / tmp[2];
			if (pdash[0] >= 0 && pdash[1] >= 0 && pdash[0] < imgsize.width && pdash[1] < imgsize.height) {
				result.at<float>(pdash[1], pdash[0]) = 1.0;
			}
		}

		return result;
	}
private:
	double voxelSideLength;
	std::unordered_map<cv::Vec3i, ScenePoint, VecHash> surfacePoints;
};
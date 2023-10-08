

class ScenePoint {
public:
	cv::Vec3f normal;
};

class RenderHelper {
public:
	RenderHelper(View& v) {
		cv::Size imgsize = v.image.size();

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

	inline void addPoint(cv::Vec3f point, cv::Vec3f normal) {
		cv::Vec3i q = point / voxelSideLength;
		ScenePoint s;
		s.normal = normal;
		surfacePoints.insert(std::make_pair(q, s));
	}

	inline cv::Vec3f addVoxelCenter(const cv::Vec3f voxel) const {
		const float center = (float)voxelSideLength * 0.5f;
		const cv::Vec3f toCenter(center, center, center);
		return voxel + toCenter;
	}

	inline cv::Vec3f getCenterOfVoxel(const cv::Vec3f point) const {
		cv::Vec3i q = point / voxelSideLength;
		return addVoxelCenter((cv::Vec3f)q * voxelSideLength);
	}

	inline cv::Vec3f centerAndRenderBack(const RenderHelper& renderHelper, const cv::Vec3f point) const {
		return renderHelper.projectPoint(getCenterOfVoxel(point));
	}

	cv::Mat directRender(View& v, bool renderNormals = false) {
		const float zfar = 1.3;
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
		auto endSurface = surfacePoints.end();

		for (auto it = surfacePoints.begin(); it != endSurface; it++) {
			cv::Vec3f p = it->first;
			p = addVoxelCenter(p * voxelSideLength);
			cv::Vec3f project = rhelper.projectPoint(p);
			cv::Vec3i pdash = project;
			if (pdash[0] >= 0 && pdash[1] >= 0 && project[2] > 0 && pdash[0] < imgsize.width && pdash[1] < imgsize.height) {
				if (renderNormals) {
					cv::Vec3f n = it->second.normal;
					n = (n + cv::Vec3f::ones() * 1.5) / 3;
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
private:
	double voxelSideLength;
	std::unordered_map<cv::Vec3i, ScenePoint, VecHash> surfacePoints;
};
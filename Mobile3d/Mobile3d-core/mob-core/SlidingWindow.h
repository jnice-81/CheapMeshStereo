#include <opencv2/opencv.hpp>
#include "View.h"
#include <vector>

class SlidingWindow {
public:
	SlidingWindow(size_t windowSize)
	{
		this->windowSize = windowSize;
		sliding_window.reserve(windowSize);
	}

	bool shouldAddImage(const cv::Mat& newExtrinsics, float minNorm, float minRot) {
		if (sliding_window.size() == 0) {
			return true;
		}

		const cv::Rect roiR = cv::Rect(0, 0, 3, 3);
		const cv::Rect roiT = cv::Rect(3, 0, 1, 3);

		View oldView = getView(0);
		cv::Mat R = newExtrinsics(roiR) * oldView.extrinsics(roiR).t();
		double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
		double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));

		cv::Mat T = newExtrinsics(roiT) - R * oldView.extrinsics(roiT);
		double normT = cv::norm(T);

		return normT >= minNorm || angle >= minRot;
	}

	void add_image(View new_view) {
		if (sliding_window.size() < windowSize) {
			sliding_window.push_back(new_view);
		}
		else {
			sliding_window[currentWrite] = new_view;
		}
		currentWrite = (currentWrite + 1) % windowSize;
	}

	inline View& getView(int idx) {
		assert(idx <= 0 && idx > -(int)sliding_window.size());

		int access = (currentWrite - 1 + idx);
		if (access < 0) {
			access = windowSize + access;
		}
		return sliding_window[access];
	}

	inline size_t size() const {
		return sliding_window.size();
	}

private:
	std::vector<View> sliding_window;
	size_t windowSize;
	size_t currentWrite = 0;
};
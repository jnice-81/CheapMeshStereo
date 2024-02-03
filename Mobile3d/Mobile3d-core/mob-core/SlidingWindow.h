#include <opencv2/opencv.hpp>
#include "View.h"
#include <vector>

class SlidingWindow {
public:
	SlidingWindow(std::size_t windowSize)
	{
		this->windowSize = windowSize;
		sliding_window.reserve(windowSize);
	}


	int add_image(View new_view) {
		if (sliding_window.size() < windowSize) {
			sliding_window.push_back(new_view);
		}
		else {
			sliding_window[currentWrite] = new_view;
		}
		currentWrite = (currentWrite + 1) % windowSize;
		currentImageIndex++;
		return currentImageIndex;
	}

	inline View& getView(const long idx) {
		assert(isIndexValid(idx));

		int access = (currentWrite - 1 + idx);
		if (access < 0) {
			access = windowSize + access;
		}
		return sliding_window[access];
	}

	inline long getCurrentImageIndex() const {
		return currentImageIndex;
	}

	inline View& getViewByImageIndex(const long imageIdx) {
		return getView(imageIdx - currentImageIndex);
	}

	inline bool isImageIndexValid(const long imageIdx) const {
		return isIndexValid(imageIdx - currentImageIndex);
	}

	inline std::size_t size() const {
		return sliding_window.size();
	}

	inline std::size_t getWindowSize() const {
		return windowSize;
	}

private:
	inline bool isIndexValid(const long idx) const {
		return idx <= 0 && idx > -(int)sliding_window.size();
	}

	std::vector<View> sliding_window;
	std::size_t windowSize;
	std::size_t currentWrite = 0;
	long currentImageIndex = -1;
};
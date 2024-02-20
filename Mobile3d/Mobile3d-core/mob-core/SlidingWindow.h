#include <opencv2/opencv.hpp>
#include "View.h"
#include <vector>

/*
A storage class storing a list of Views with fixed length
*/
class SlidingWindow {
public:
	/*
	windowSize: The maximum allowed amount of elements
	*/
	SlidingWindow(std::size_t windowSize)
	{
		this->windowSize = windowSize;
		sliding_window.reserve(windowSize);
	}

	/*
	Add a view to the slidingwindow.
	new_view: The view to add
	returns: The id which was associated with the view
	*/
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

	/*
	Get a View.
	idx: The index is a value between 0 and - size of the sliding window. 0 indicates to retrieve the last added image,
		while - size of the sliding window + 1 indicates to get the "oldest" image
	*/
	inline View& getView(const long idx) {
		assert(isIndexValid(idx));

		int access = (currentWrite - 1 + idx);
		if (access < 0) {
			access = windowSize + access;
		}
		return sliding_window[access];
	}

	/*
	Get the id which was associated with the newest view.
	*/
	inline long getCurrentImageIndex() const {
		return currentImageIndex;
	}

	/*
	Get a view via it's id.
	*/
	inline View& getViewByImageIndex(const long imageIdx) {
		return getView(imageIdx - currentImageIndex);
	}

	/*
	Check if an image id can be safely accessed. (I.e. the image still exists in the sliding window)
	*/
	inline bool isImageIndexValid(const long imageIdx) const {
		return isIndexValid(imageIdx - currentImageIndex);
	}

	/*
	Get the current size of the window. Can be between 0 and the maximum windowSize.
	*/
	inline std::size_t size() const {
		return sliding_window.size();
	}

	/*
	Get the window size associated with this sliding window.
	*/
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

#ifdef DEBUG
void overlay(cv::Mat depth, cv::Mat img, float imgAlpha = 0.1) {
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	img.convertTo(img, CV_32F, (1.0 / 255) * imgAlpha);

	cv::Mat overlayed = img + depth;
	cv::resize(overlayed, overlayed, cv::Size(), 0.5, 0.5);

	cv::imshow("overlayed", overlayed);
	cv::waitKey(0);
}
#endif // DEBUG

class MsClock {
public:
	MsClock() {
		reset();
	}

	void reset() {
		startTime = std::chrono::system_clock::now();
	}

	void printAndReset(std::string msg) {
		std::cout << msg << " " << 
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count() << std::endl;
		reset();
	}

	std::chrono::system_clock::time_point startTime;
};

template<typename T> inline T vecZeros() {
	auto g = T::zeros();
	return T(g.val);
}

template<typename T> inline T vecOnes() {
	auto g = T::ones();
	return T(g.val);
}
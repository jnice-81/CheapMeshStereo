

void overlay(cv::Mat depth, cv::Mat img) {
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	img.convertTo(img, CV_32F, 1.0 / 255);

	cv::Mat overlayed = img + depth;
	cv::resize(overlayed, overlayed, cv::Size(), 0.5, 0.5);

	cv::imshow("overlayed", overlayed);
	cv::waitKey(0);
}
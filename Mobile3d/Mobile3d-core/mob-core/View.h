class View {
public:
    View(cv::Mat image, cv::Mat intrinsics, cv::Mat extrinsics)
    {
        this->image = image;
        this->intrinsics = intrinsics;
        this->extrinsics = extrinsics;
    }

    cv::Mat image;
    cv::Mat_<double> intrinsics;
    cv::Mat_<double> extrinsics;
};
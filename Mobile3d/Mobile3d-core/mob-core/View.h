class View {
public:
    View(cv::Mat image, cv::Mat intrinsics, cv::Mat extrinsics, std::unordered_set<int> keyPointIds)
    {
        this->image = image;
        this->intrinsics = intrinsics;
        this->extrinsics = extrinsics;
        this->keyPointIds = keyPointIds;
    }

    cv::Mat image;
    cv::Mat_<double> intrinsics;
    cv::Mat_<double> extrinsics;
    std::unordered_set<int> keyPointIds;
};
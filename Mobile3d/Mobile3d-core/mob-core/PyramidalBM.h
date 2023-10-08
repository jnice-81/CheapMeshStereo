class PyramidalBM {
public:
    static cv::Mat compute(
        cv::Mat left,
        cv::Mat right,
        int minSearch = -150,
        int maxSearch = 150,
        int numLevels = 3,
        int blockWidth = 32,
        int blockHeight = 8,
        float rejectMaxError = 0.5,
        float minRatio = 2.0
    ) {
        assert((blockWidth >> numLevels) << numLevels == blockWidth);
        assert((blockHeight >> numLevels) << numLevels == blockHeight);
        auto imageSize = left.size();
        assert((imageSize.width >> numLevels) << numLevels == imageSize.width);
        assert((imageSize.height >> numLevels) << numLevels == imageSize.height);

        cv::Mat result = cv::Mat::zeros(left.size(), CV_16S);

        for (int level = numLevels; level >= 0; level--) {
            
        }
    }
};
class PyramidalBM {
public:
    cv::Mat compute(
        cv::Mat left, 
        cv::Mat right, 
        int minSearch = -160,
        int maxSearch = 160,
        int numLevel = 2,
        int blockWidth = 8,
        int blockHeight = 4,
        int border = 0,
        float rejectMaxError = 0.5,
        float minRatio = 2.0) {
            
        auto imageSize = left.size();
        int level = numLevel;
        assert((blockWidth >> level) << level == blockWidth);
        assert((blockHeight >> level) << level == blockHeight);

        level -= 1;

        assert((imageSize.width >> level) << level == imageSize.width);
        assert((imageSize.height >> level) << level == imageSize.height);
        assert(left.type() == CV_8UC3 && right.type() == CV_8UC3);
        assert(left.size() == right.size());


        cv::Mat previousDisparity;
        for (int currentLevel = level; level >= 0; level--) {
            int scaleFactor = 1 << currentLevel;
            cv::Mat leftScaled;
            cv::resize(left, leftScaled, imageSize / scaleFactor);
            cv::Mat rightScaled;
            cv::resize(right, rightScaled, imageSize / scaleFactor);

            cv::Mat disp = computeLevel(
                leftScaled,
                rightScaled,
                previousDisparity,
                minSearch >> currentLevel,
                maxSearch >> currentLevel,
                blockWidth >> currentLevel,
                blockHeight >> currentLevel,
                border >> currentLevel,
                rejectMaxError,
                minRatio);
            previousDisparity = disp;
        }

        return previousDisparity;
    }

private:
    cv::Mat computeLevel(
        cv::Mat left,
        cv::Mat right, 
        const cv::Mat previousdisparity,
        int minSearch,
        int maxSearch,
        int blockWidth,
        int blockHeight,
        int border,
        float rejectMaxError,
        float minRatio) {

        auto imgsize = left.size();
        cv::Mat result = cv::Mat::zeros(imgsize, CV_16S);
        bool isPrevious = !previousdisparity.empty();

        for (int x = border; x < imgsize.width - border; x++) {
            for (int y = 0; y < imgsize.height - blockHeight; y++) {
                float minVal;
                float secondMinVal;

                int adaptedMin = minSearch;
                int adaptedMax = maxSearch;
                if(isPrevious) {
                    int dispX = x / 2;
                    int dispY = y / 2;
                    int min = 100000;
                    int max = -100000;

                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            int idxX = dispX -1 + i;
                            int idxY = dispY -1 + i;
                            if (idxX >= 0 && idxY >= 0 && idxX < imgsize.width / 2 && idxY < imgsize.height / 2) {
                                short h = previousdisparity.at<short>(idxX, idxY);
                                if (h < min) {
                                    min = h;
                                }
                                if (h > max) {
                                    max = h;
                                }
                            }
                        }
                    }
                }


                int fromX = x + minSearch;
                if (fromX < 0) {
                    fromX = 0;
                }
                int toX = x + maxSearch;
                if (toX >= imgsize.width - blockWidth) {
                    toX = imgsize.width - blockWidth;
                }

                for (int t = fromX; t < toX; t++) {

                }

            }
        }
        
        return result;
    }
};
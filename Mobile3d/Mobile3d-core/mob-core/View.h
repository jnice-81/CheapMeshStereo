#pragma once

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

    static cv::Mat_<double> oglExtrinsicsToCVExtrinsics(const cv::Mat &extrinsics) {
        cv::Mat_<double> ogl_to_cv = cv::Mat_<double>(4, 4);
        ogl_to_cv <<
                  1, 1, 1, 1,
                -1, -1, -1, -1,
                -1, -1, -1, -1,
                0, 0, 0, 1;

        return extrinsics.mul(ogl_to_cv);
    }

    static cv::Mat_<double> oglIntrinsicsToCVIntrinsics(const cv::Mat &intrinsics, const cv::Size imgsize) {
        cv::Mat_<double> cvintrinsics = cv::Mat_<double>(3, 3);
        cvintrinsics <<
                     intrinsics.at<double>(0, 0) * 0.5 * imgsize.width, 0, intrinsics.at<double>(0, 2) + imgsize.width * 0.5,
                0, intrinsics.at<double>(1, 1) * 0.5 * imgsize.height, intrinsics.at<double>(1, 2) + imgsize.height * 0.5,
                0, 0, 1;
        return cvintrinsics;
    }
};
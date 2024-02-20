#pragma once

/*
A class storing an image, it's associated intrinsics and extrinsics, both in opencv format.
*/
class View {
public:
    View(cv::Mat image, cv::Mat intrinsics, cv::Mat extrinsics)
    {
        this->image = image;
        this->intrinsics = intrinsics;
        this->extrinsics = extrinsics;
    }

    View()
    {

    }

    cv::Mat image;
    cv::Mat_<double> intrinsics;
    cv::Mat_<double> extrinsics;


    /*
    Convenience function converting standard opengl extrinsics to opencv extrinsics.
    */
    static cv::Mat_<double> oglExtrinsicsToCVExtrinsics(const cv::Mat &extrinsics) {
        cv::Mat_<double> ogl_to_cv = cv::Mat_<double>(4, 4);
        ogl_to_cv <<
                  1, 1, 1, 1,
                -1, -1, -1, -1,
                -1, -1, -1, -1,
                0, 0, 0, 1;

        return extrinsics.mul(ogl_to_cv);
    }

    /*
    Convenience function converting standard opengl intrinsics to opencg intrinsics.
    */
    static cv::Mat_<double> oglIntrinsicsToCVIntrinsics(const cv::Mat &intrinsics, const cv::Size imgsize) {
        cv::Mat_<double> cvintrinsics = cv::Mat_<double>(3, 3);
        cvintrinsics <<
                     intrinsics.at<double>(0, 0) * 0.5 * imgsize.width, 0, intrinsics.at<double>(0, 2) + imgsize.width * 0.5,
                0, intrinsics.at<double>(1, 1) * 0.5 * imgsize.height, intrinsics.at<double>(1, 2) + imgsize.height * 0.5,
                0, 0, 1;
        return cvintrinsics;
    }

    /*
    Get the relative rotation and translation between two extrinsics matrices.
    */
    static std::pair<float, float> getRelativeRotationAndTranslation(const cv::Mat& extrinsics1, const cv::Mat &extrinsics2) {
        const cv::Rect roiR = cv::Rect(0, 0, 3, 3);
        const cv::Rect roiT = cv::Rect(3, 0, 1, 3);

        cv::Mat R = extrinsics1(roiR) * extrinsics2(roiR).t();
        double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
        double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));

        cv::Mat T = extrinsics1(roiT) - R * extrinsics2(roiT);
        double normT = cv::norm(T);

        return std::make_pair(angle, normT);
    }
};
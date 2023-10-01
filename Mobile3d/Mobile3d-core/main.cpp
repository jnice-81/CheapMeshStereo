#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <list>

using jfile = nlohmann::json;

class View {
public:
    View(std::string image_path, cv::Mat image, cv::Mat intrinsics, cv::Mat extrinsics, std::unordered_set<int> keyPointIds)
    {
        this->image_path = image_path;
        this->image = image;
        this->intrinsics = intrinsics;
        this->extrinsics = extrinsics;
        this->keyPointIds = keyPointIds;
    }

    void update_image(cv::Mat image) {
        this->image = image;
    }

    void release_image() {
        image.release();
    }

    std::string image_path;
    cv::Mat image;
    cv::Mat intrinsics;
    cv::Mat extrinsics;
    std::unordered_set<int> keyPointIds;
};

cv::Mat load_matrix_from_json(nlohmann::json jarray) {
    cv::Mat matrix(4, 4, CV_64F);

    int row = 0;
    int col = 0;
    for (const auto& value : jarray) {
        double t = value.get<double>();
        matrix.at<double>(row, col) = t;
        row++;
        if (row == 4) {
            row = 0;
            col++;
        }
    }

    return matrix;
}

cv::Mat intrinsics_to_opencv_intrinsics(cv::Mat& input) {
    cv::Mat_<double> v1_intrinsics_opencv(3, 3);
    double c = 1;
    v1_intrinsics_opencv <<
        910.3 * c, 0, 358.19,
        0, 905.35 * c, 639.5,
        0, 0, 1;

    return v1_intrinsics_opencv;
}

int main()
{
    const std::string base_dir = "C:\\Users\\Admin\\Desktop\\Mobile3d\\Mobile3d\\3dmodels";
    const std::string project_name = "gerade";
    const std::string projectfolder = base_dir + "\\" + project_name;

    std::ifstream arcorefile(projectfolder + "\\ARCoreData.json");
    auto arcore = jfile::parse(arcorefile);

    std::vector<View> views;

    for (const auto& obj : arcore["ARCoreData"]) {
        std::string name = obj["name"];
        cv::Mat intrinsics = load_matrix_from_json(obj["projection"]);
        cv::Mat extrinsics = load_matrix_from_json(obj["viewmatrix"]);
        std::unordered_set<int> keyPointIds;
        for (const auto& k : obj["pointIDs"]) {
            keyPointIds.insert(k.get<int>());
        }
        std::string imgpath = projectfolder + "\\images\\" + name + ".jpg";
        cv::Mat image = cv::imread(imgpath);
        views.emplace_back(imgpath, image, intrinsics, extrinsics, keyPointIds);
    }

    View v1 = views[0];
    View v2 = views[1];

    cv::Mat inv1, inv2;
    cv::Mat R1, R2, T1, T2;
    cv::invert(v1.extrinsics, inv1);
    cv::invert(v2.extrinsics, inv2);
    cv::Rect roiR = cv::Rect(0, 0, 3, 3);
    cv::Rect roiT = cv::Rect(3, 0, 1, 3);
    R1 = inv1(roiR);
    R2 = inv2(roiR);
    T1 = inv1(roiT);
    T2 = inv2(roiT);
    cv::Mat R = R1 * R2;
    cv::Mat T = T1 - T2;

    //std::cout << inv1 << std::endl << inv2 << std::endl << R1 << std::endl << R2 << std::endl
    //    << T1 << std::endl << T2;

    /*
    cv::Mat R = R2 * R1.t();
    cv::Mat T = T2 - R * T1;

    R = cv::Mat::eye(3, 3, CV_64F);
    T = cv::Mat::ones(3, 1, CV_64F) * 10000000;

    auto sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> v1kp;
    std::vector<cv::KeyPoint> v2kp;
    cv::Mat v1des;
    cv::Mat v2des;
    sift->detectAndCompute(v1.image, cv::Mat(), v1kp, v1des);
    sift->detectAndCompute(v2.image, cv::Mat(), v2kp, v2des);

    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(v1des, v2des, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < 0.15 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // Extract the keypoints for good matches
    std::vector<cv::Point2f> pts1, pts2;
    for (const cv::DMatch& match : good_matches) {
        pts1.push_back(v1kp[match.queryIdx].pt);
        pts2.push_back(v2kp[match.trainIdx].pt);
    }

    //cv::Mat matchdraw;
    //cv::drawMatches(v1.image, v1kp, v2.image, v2kp, good_matches, matchdraw);
    //cv::imshow("hasdl", matchdraw);
    //cv::waitKey(0);

    cv::Mat fundamental_matrix;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC);
    */

    cv::Mat rR1, rR2, rP1, rP2, Q;
    auto img_size = v1.image.size();

    cv::stereoRectify(intrinsics_to_opencv_intrinsics(v1.intrinsics), cv::Mat(), intrinsics_to_opencv_intrinsics(v2.intrinsics), cv::Mat(), img_size,
        R, T, rR1, rR2, rP1, rP2, Q, cv::CALIB_ZERO_DISPARITY, 0);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(intrinsics_to_opencv_intrinsics(v1.intrinsics), cv::Mat(), rR1, rP1, img_size, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(intrinsics_to_opencv_intrinsics(v2.intrinsics), cv::Mat(), rR2, rP2, img_size, CV_32FC1, map2x, map2y);

    cv::Mat rectified_image1, rectified_image2;
    cv::remap(v1.image, rectified_image1, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(v2.image, rectified_image2, map2x, map2y, cv::INTER_LINEAR);

    cv::imshow("v1", rectified_image1);
    cv::imshow("v2", rectified_image2);
    cv::imshow("v1src", v1.image);
    cv::imshow("v2src", v2.image);
    cv::waitKey(0);

}
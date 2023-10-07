#include <json.hpp>
#include "mob-core/Reconstruct.h"

#define LINUX

using jfile = nlohmann::json;

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

int main()
{
    #ifdef LINUX
        const std::string base_dir = "/home/jannis/Schreibtisch/Mobile3d/3dmodels/";
    #elif WINDOWS
        const std::string base_dir = "whatever";
    #endif
    const std::string project_name = "old/corner";
    const std::string projectfolder = base_dir + "/" + project_name;

    std::ifstream arcorefile(projectfolder + "/ARCoreData.json");
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
        #ifdef LINUX
            std::string imgpath = projectfolder + "/images/" + name + ".jpg";
        #elif WINDOWS
            std::string imgpath = projectfolder + "\\images\\" + name + ".jpg";
        #endif
        cv::Mat image = cv::imread(imgpath);
        views.emplace_back(image, intrinsics, extrinsics, keyPointIds);
    }

    View v1 = views[8];
    View v2 = views[9];

    Scene gm(0.001);
    Reconstruct g(gm);

    g.OpenGL2OpenCVView(v1);
    g.OpenGL2OpenCVView(v2);
    g.add_image(v1);
    g.add_image(v2);
}
#include <json.hpp>

#define WINDOWS
//#define LINUX
#include "mob-core/Reconstruct.h"

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
    #endif
    #ifdef WINDOWS
        const std::string base_dir = "C:/Users/Admin/Desktop/Mobile3d/3dmodels";
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
        std::string imgpath = projectfolder + "/images/" + name + ".jpg";
        cv::Mat image = cv::imread(imgpath);
        views.emplace_back(image, intrinsics, extrinsics, keyPointIds);
    }

    Scene gm(0.01);
    Reconstruct g(gm);

    //for (int i = views.size()-1; i >= views.size()-2; i--) {
    /*
    for (int i = 0; i < views.size(); i++) {
        g.OpenGL2OpenCVView(views[i]);
        g.add_image(views[i]);
    }*/

    gm.import_xyz("h.xyz");
    gm.filterOutliers(10, 200);
    PoissonSurfaceReconstruct<int, float, 3>::reconstructAndExport(gm, "exp.ply", 10); 
    

    //gm.export_xyz("h.xyz");
    //cv::Mat out = gm.directRender(views[8], false);
    //cv::resize(out, out, cv::Size(), 0.8, 0.8);
    //cv::imshow("q", out);
    //cv::waitKey(0);
    
    //overlay(out, views[8].image, 0.5);

    return 0;
}
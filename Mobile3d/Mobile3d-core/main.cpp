#include <json.hpp>

#define WINDOWS
//#define LINUX
#include <opencv2/highgui.hpp>
#include "mob-core/Reconstruct.h"
#include "mob-core/SlidingWindow.h"
//#include "mob-core/PoissonSurfaceReconstruct.h"

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
    const std::string project_name = "imagestream/chair";
    const std::string projectfolder = base_dir + "/" + project_name;

    std::ifstream arcorefile(projectfolder + "/ARCoreData.json");
    auto arcore = jfile::parse(arcorefile);

    Scene<3, bool> gm(0.02, std::vector<int>({ 10, 20, 2 }));
    SlidingWindow slideWindow(10);

    int nimg = 0;
    for (const auto& obj : arcore["ARCoreData"]) {
        std::string name = obj["name"];
        cv::Mat extrinsics = View::oglExtrinsicsToCVExtrinsics(load_matrix_from_json(obj["viewmatrix"]));
        /*std::unordered_set<int> keyPointIds;
        for (const auto& k : obj["pointIDs"]) {
            keyPointIds.insert(k.get<int>());
        }*/
        nimg++;
        if (slideWindow.shouldAddImage(extrinsics, 0.05, 0.1)) {
            std::string imgpath = projectfolder + "/images/" + name + ".jpg";
            std::cout << imgpath << std::endl;
            cv::Mat image = cv::imread(imgpath);
            cv::Mat intrinsics = View::oglIntrinsicsToCVIntrinsics(load_matrix_from_json(obj["projection"]), image.size());
            View v(image, intrinsics, extrinsics);
            slideWindow.add_image(v);
            
            if (slideWindow.size() >= 5) {
                std::vector<ScenePoint> v;
                for (int i = 1; i < 5; i++) {
                    Reconstruct::compute3d(slideWindow.getView(0), slideWindow.getView(-i), v, 0.5, 5, 0.02);
                }
                for (const auto& s : v) {
                    gm.addPoint(s);
                }
            }
        }
    }

    
    gm.export_xyz("h.xyz");

    return 0;
}
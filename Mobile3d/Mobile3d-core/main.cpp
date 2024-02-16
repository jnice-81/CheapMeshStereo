#include <json.hpp>

#define WINDOWS
//#define LINUX
#include <opencv2/highgui.hpp>
#include "mob-core/Reconstruct.h"
#include "mob-core/SlidingWindow.h"
//#include "mob-core/PoissonSurfaceReconstruct.h"
#include "mob-core/Scene.h"
#include "mob-core/SurfaceReconstruct.h"

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

void surface_test() {
    Scene<3, bool> gm(0.03, std::vector<int>({ 5, 5, 4 }));
    gm.import_xyz("h.xyz");

    SurfaceReconstruct r(0.04, 4, 3.0f);

    std::cout << "Starting" << std::endl;
    MsClock c;
    r.computeSurface(gm);
    c.printAndReset("Ended");
    //r.exportImplNorm.export_xyz("samples.xyz");
    r.exportObj("a.obj");
}


void generate_test() {
#ifdef LINUX
    const std::string base_dir = "/home/jannis/Schreibtisch/Mobile3d/3dmodels/";
#endif
#ifdef WINDOWS
    const std::string base_dir = "C:/Users/Admin/Desktop/Mobile3d/3dmodels";
#endif

    Scene<3, bool> gm(0.02, std::vector<int>({ 10, 20, 2 }));

    const std::string project_name = "imagestream/chair";
    const std::string projectfolder = base_dir + "/" + project_name;

    std::ifstream arcorefile(projectfolder + "/ARCoreData.json");
    auto arcore = jfile::parse(arcorefile);

    SlidingWindow slideWindow(10);
    std::vector<ScenePoint> tmp;
    int nimg = 0;
    for (const auto& obj : arcore["ARCoreData"]) {
        std::string name = obj["name"];
        cv::Mat extrinsics = View::oglExtrinsicsToCVExtrinsics(load_matrix_from_json(obj["viewmatrix"]));
        nimg++;
        std::pair<float, float> rel;
        if (slideWindow.size() > 0) {
            rel = View::getRelativeRotationAndTranslation(slideWindow.getView(0).extrinsics, extrinsics);
        }
        if (slideWindow.size() == 0 || rel.second >= 0.05 || rel.first >= 0.1) {
            std::string imgpath = projectfolder + "/images/" + name + ".jpg";
            std::cout << imgpath << std::endl;
            cv::Mat image = cv::imread(imgpath);
            cv::Mat intrinsics = View::oglIntrinsicsToCVIntrinsics(load_matrix_from_json(obj["projection"]), image.size());
            View v(image, intrinsics, extrinsics);
            slideWindow.add_image(v);

            if (slideWindow.size() >= 5) {
                for (int i = 1; i < 5; i++) {
                    Reconstruct::compute3d(slideWindow.getView(0), slideWindow.getView(-i), tmp, 0.5, 5, 0.02);
                    gm.addAllSingleCount(tmp);
                    gm.filterNumviews(2, tmp);
                    tmp.clear();
                }
            }
        }
    }
    //gm.filterNumviews(2);

    gm.export_xyz("h.xyz");
}

int main()
{
    surface_test();

    return 0;
}
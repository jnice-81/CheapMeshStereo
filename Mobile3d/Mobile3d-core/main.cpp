#include <json.hpp>

#define WINDOWS
//#define LINUX
#include <opencv2/highgui.hpp>
#include "mob-core/Reconstruct.h"
#include "mob-core/SlidingWindow.h"
#include "mob-core/Scene.h"
#include "mob-core/SurfaceReconstruct.h"

/*
Only for testing out stuff
*/

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
    Scene<1> gm(0.08, std::vector<int>({ 4 }));
    std::string base = "C:/Users/Admin/Desktop/";
    gm.import_xyz(base + "out.xyz");

    double mean = 0.0;
    for (auto it = gm.surfacePoints.treeIteratorBegin(); !it.isEnd(); it++) {
        mean += (double)it->second.numhits;
    }
    mean /= gm.surfacePoints.getPointCount();
    std::cout << "Mean hits " << mean << std::endl;

    gm.normalizeNormals();
    SurfaceReconstruct<0, 1> r(gm, 5, 3.5f, -1);

    std::cout << "Starting" << std::endl;
    MsClock c;
    r.computeSurface();
    c.printAndReset("Ended");
    r.exportObj(base + "a.obj");
}


void generate_test() {
#ifdef LINUX
    const std::string base_dir = "/home/jannis/Schreibtisch/Mobile3d/3dmodels/";
#endif
#ifdef WINDOWS
    const std::string base_dir = "C:/Users/Admin/Desktop/Mobile3d/3dmodels";
#endif

    Scene<3> gm(0.02, std::vector<int>({ 10, 20, 2 }));

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
                    tmp.clear();
                }
            }
        }
    }
    gm.filterNumviews(2);

    gm.export_xyz("h.xyz");
}

/*
void external_surface(int argc, char* argv[]) {

    std::string ipath = argv[1];
    std::string opath = argv[2];
    
    float voxelsize = std::stof(argv[3]);
    int svoxelcount = std::stoi(argv[4]);

    float recvoxelsize = std::stof(argv[5]);
    float minweight = std::stof(argv[6]);
    float scale = std::stof(argv[7]);

    Scene<1> g(voxelsize, std::vector<int>({ svoxelcount }));
    SurfaceReconstruct m(recvoxelsize, minweight, scale);

    g.import_xyz(ipath);
    m.computeSurface(g);
    m.exportObj(opath);
    std::cout << "Finished reconstruction" << std::endl;
}*/

int main(int argc, char* argv[])
{
    //generate_test();
    surface_test();

    return 0;
}
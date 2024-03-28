#include "../CheapMeshStereo-core/CheapMeshStereoCore/helpers.h"
#include "../CheapMeshStereo-core/CheapMeshStereoCore/SurfaceReconstruct.h"
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <input> <output> <granularity> <scale> <minweight> <minviewfilter> <bias>" << std::endl;
        std::cerr << std::endl
            << "input: path to an xyz file with normals." << std::endl
            << "output: path to an obj file which will be generated (carefull: overwrittes without warning)." << std::endl
            << "granularity: How large the sidelength of a voxel is which can contain at most one point. "
            << "Small values (as long as larger than the actual point cloud precision) give detailed but slow reconstruction with large output. "
            << "Analogue but reversed for large values." << std::endl
            << "scale: How large is the region around a point where it influences surface generation? Something like 2.0 or 3.0 is usually a good choice. "
            << "Note: This is multiplied with the granularity." << std::endl
            << "minweight: The minimum weight of the implicit function for a surface to be extracted." << std::endl
            << "minviewfilter: Prefilter points to have at least so many observations. An observation is defined as one point which is lying in "
            << "a voxel. You can also influence this value by scaling the normal of your point. Note that this is essentially always bad for quality"
            << "(use minweight against outliers), but might be good for speed in large scenes with lots and lots of noise." << std::endl
            << "bias: If positive use dual contouring and adapt vertices such that they fit normals, imposing the defined bias towards the center point. "
            << "If negative simply directly use the center point (a lot faster; especially for high resolutions the influence is small).";
        return 1;
    }

    std::string input = argv[1];
    std::string output = argv[2];
    float granularity = std::stof(argv[3]);
    float scale = std::stof(argv[4]);
    float minweight = std::stof(argv[5]);
    int minviewfilter = std::stoi(argv[6]);
    float bias = std::stof(argv[7]);
    int NormalCacheSize;
    if (bias < 0) {
        NormalCacheSize = 0;
    }
    else {
        NormalCacheSize = 10000;
    }
    // Essentially pick a reasonable size for selection. Reasonable is something where the voxel size is somwhere
    // between scale and 2*scale + a bit of overhead. So picked the middle here.
    int supersize = static_cast<int>(std::ceil(scale * 1.5));

    MsClock clock;

    Scene<1> s(granularity, std::vector<int>({ supersize }));
    s.import_xyz(input);

    if (minviewfilter > 0) {
        s.filterNumviews(minviewfilter);
    }

    clock.printAndReset("Loaded pointcloud");
    std::cout << "Contains " << s.surfacePoints.getPointCount() << " voxels" << std::endl;

    SurfaceReconstruct<0, 1> r(s, minweight, scale, 20000, bias, NormalCacheSize);
    r.computeSurface();

    clock.printAndReset("Computed Surface");

    r.exportObj(output);

    clock.printAndReset("Converting and Saving Mesh");

    return 0;
}
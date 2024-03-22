# CheapMeshStereo
This repository contains two things: First a real time posed image to mesh pipeline, written in c++ using only CPU, consisting of three semiindependent modules: Stereo Vision, Point cloud representation and Surface extraction. Secondly and a sample android APP which uses the pipeline to generate meshes of scenes scanned with the camera.

For now only the point cloud collection and visualization is done in realtime, the mesh generation is done in a later step. Time depends on the scene size and point cloud resolution, but is usually far below one minute.

To see some results checkout https://drive.google.com/drive/folders/1-nPmOGR-nMf234L9wkdjvAOtcNwJ7jJG?usp=sharing

## Whom could this be interesting for

First things first: The pipeline focuses on efficiency, less so on quality, so this is not the right thing if you are interested in nice looking 3d models (more so if you require 3d models for technical tasks). 
Also the purpose of the App was to test/showcase the pipeline, it is definitively not end user friendly. 

Apart from that:

1. Well if you are interested in a fast and portable image to mesh pipeline ;). If quality is of more concern than performance take a look at [OpenMVS](https://github.com/cdcseacave/openMVS)
2. Point cloud storage. There are large libraries for dealing with point clouds, e.g. [PCL](https://github.com/PointCloudLibrary/pcl). Could still be interesting if:
    - Aim is to (swiftly) downsample a high resolution point cloud to speed up downstream tasks
    - Your situation is similar to the one here, i.e. there is some process generating/updating a point cloud and there is a need for fast updating while always being able to query the current points/local regions of space efficiently
3. Mesh reconstruction from point clouds. [Poisson Surface Reconstruction](https://github.com/mkazhdan/PoissonRecon) is far more mature.
    - You simply want a nice, noise free mesh from an oriented point cloud: Use poisson surface reconstruction/something else than this
    - If speed is required, quality is less of a concern: Probably downsample the pointcloud with the point cloud represenation here, then use poisson surface reconstruction. However this is also 
		pretty fast and if loading/storing would be optimized it may outperform the above approach.
    - The fact that poisson surface reconstruction "hallucinates" surface on holes is a large issue: This implementation could be interesting
    - The mesh should be efficiently updated given a changing point cloud. Not supported for now, but it should be relatively easy to extend it towards it.
  
## License

MIT, except for everything in CheapMeshStereo\SampleApp which is licensed under Apache 2. Reason is that this was built using the hello_arcore sample app by google as a starting point.
 
## How to use

The pipeline heavily relies on OpenCV. It was developed and tested using OpenCV 4.8. 

Apart from that, it's just header files so simply include and your good to go. You could take a quick look at TODO ADD REPORT. 
The code of the pipeline is in CheapMeshStereo/CheapMeshStereo-core/CheapMeshStereoCore. All functions/classes are commented. 

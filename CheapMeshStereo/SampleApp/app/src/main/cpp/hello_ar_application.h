/*
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef C_ARCORE_HELLOE_AR_HELLO_AR_APPLICATION_H_
#define C_ARCORE_HELLOE_AR_HELLO_AR_APPLICATION_H_

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <android/asset_manager.h>
#include <jni.h>

#include <memory>
#include <set>
#include <string>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>

#include "arcore_c_api.h"
#include "background_renderer.h"
#include "glm.h"
#include "obj_renderer.h"
#include "plane_renderer.h"
#include "point_cloud_renderer.h"
#include "dense_point_renderer.h"
#include "texture.h"
#include "util.h"

#include <Reconstruct.h>
#include <SlidingWindow.h>
#include <dense_point_renderer.h>
#include <SurfaceReconstruct.h>

namespace hello_ar {

// HelloArApplication handles all application logics.
class HelloArApplication {
 public:
  // Constructor and deconstructor.
  explicit HelloArApplication(AAssetManager* asset_manager);
  ~HelloArApplication();

  // OnPause is called on the UI thread from the Activity's onPause method.
  void OnPause();

  // OnResume is called on the UI thread from the Activity's onResume method.
  void OnResume(JNIEnv* env, void* context, void* activity);

  // OnSurfaceCreated is called on the OpenGL thread when GLSurfaceView
  // is created.
  void OnSurfaceCreated();

  // OnDisplayGeometryChanged is called on the OpenGL thread when the
  // render surface size or display rotation changes.
  //
  // @param display_rotation: current display rotation.
  // @param width: width of the changed surface view.
  // @param height: height of the changed surface view.
  void OnDisplayGeometryChanged(int display_rotation, int width, int height);

  // OnDrawFrame is called on the OpenGL thread to render the next frame.
  void OnDrawFrame(bool depthColorVisualizationEnabled,
                   bool useDepthForOcclusion);

  void ComputeSurface();

  void ChangeGranularity(float granularity);

 private:
  glm::mat3 GetTextureTransformMatrix(const ArSession* session,
                                      const ArFrame* frame);
  ArSession* ar_session_ = nullptr;
  ArFrame* ar_frame_ = nullptr;

  bool install_requested_ = false;
  int width_ = 1;
  int height_ = 1;
  int display_rotation_ = 0;

  AAssetManager* const asset_manager_;

  // The anchors at which we are drawing android models using given colors.
  struct ColoredAnchor {
    ArAnchor* anchor;
    ArTrackable* trackable;
    float color[4];
  };

  std::vector<ColoredAnchor> anchors_;

  PointCloudRenderer point_cloud_renderer_;
  BackgroundRenderer background_renderer_;
  dense_point_renderer densePointRenderer_;

  Scene<3> collectedScene;
  SlidingWindow slideWindow;
  std::vector<std::pair<std::size_t, std::size_t>> bufferedComputations;
  std::vector<std::vector<ScenePoint>> reconstructorOutput;
  std::list<std::vector<ScenePoint>> updatedPointsForRender;
  std::vector<std::future<void>> reconstructionFuture;
  std::future<cv::Mat> imgloadFuture;
  int dbgidx = 0;
  std::fstream dbgexport;
  volatile bool stopFutureComputations = false;
  GLubyte* pixels;

  void ConfigureSession();
};
}  // namespace hello_ar

#endif  // C_ARCORE_HELLOE_AR_HELLO_AR_APPLICATION_H_

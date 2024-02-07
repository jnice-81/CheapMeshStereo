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

#include "hello_ar_application.h"

#include <android/asset_manager.h>

#include <array>

#include "arcore_c_api.h"
#include "plane_renderer.h"
#include "util.h"
#include <GLES3/gl31.h>

#include <android/log.h>

//#include "PoissonSurfaceReconstruct.h"

namespace hello_ar {

HelloArApplication::HelloArApplication(AAssetManager* asset_manager)
    : asset_manager_(asset_manager), collectedScene(0.03, std::vector<int>({10, 20, 2})),
      slideWindow(20) {}

HelloArApplication::~HelloArApplication() {
  if (ar_session_ != nullptr) {
    ArSession_destroy(ar_session_);
    ArFrame_destroy(ar_frame_);
  }
}

void HelloArApplication::OnPause() {
  LOGI("OnPause()");
  if (ar_session_ != nullptr) {
    ArSession_pause(ar_session_);
  }
}

void HelloArApplication::OnResume(JNIEnv* env, void* context, void* activity) {
  LOGI("OnResume()");

  if (ar_session_ == nullptr) {
    ArInstallStatus install_status;
    // If install was not yet requested, that means that we are resuming the
    // activity first time because of explicit user interaction (such as
    // launching the application)
    bool user_requested_install = !install_requested_;

    // === ATTENTION!  ATTENTION!  ATTENTION! ===
    // This method can and will fail in user-facing situations.  Your
    // application must handle these cases at least somewhat gracefully.  See
    // HelloAR Java sample code for reasonable behavior.
    CHECKANDTHROW(
        ArCoreApk_requestInstall(env, activity, user_requested_install,
                                 &install_status) == AR_SUCCESS,
        env, "Please install Google Play Services for AR (ARCore).");

    switch (install_status) {
      case AR_INSTALL_STATUS_INSTALLED:
        break;
      case AR_INSTALL_STATUS_INSTALL_REQUESTED:
        install_requested_ = true;
        return;
    }

    // === ATTENTION!  ATTENTION!  ATTENTION! ===
    // This method can and will fail in user-facing situations.  Your
    // application must handle these cases at least somewhat gracefully.  See
    // HelloAR Java sample code for reasonable behavior.
    CHECKANDTHROW(ArSession_create(env, context, &ar_session_) == AR_SUCCESS,
                  env, "Failed to create AR session.");

    ConfigureSession();
    ArFrame_create(ar_session_, &ar_frame_);

    ArSession_setDisplayGeometry(ar_session_, 0, 720,
                                 1280);
  }

  const ArStatus status = ArSession_resume(ar_session_);
  CHECKANDTHROW(status == AR_SUCCESS, env, "Failed to resume AR session.");
}

void HelloArApplication::OnSurfaceCreated() {
  LOGI("OnSurfaceCreated()");

  background_renderer_.InitializeGlContent(asset_manager_);
  point_cloud_renderer_.InitializeGlContent(asset_manager_);
  densePointRenderer_.InitializeGLContent();
}

void HelloArApplication::OnDisplayGeometryChanged(int display_rotation,
                                                  int width, int height) {
  LOGI("OnSurfaceChanged(%d, %d, %d)", width, height, display_rotation);
  glViewport(0, 0, width, height);
  display_rotation_ = display_rotation;
  width_ = width;
  height_ = height;
}

inline cv::Mat glm4x4ToCvMat(glm::mat4 a) {
    cv::Mat b(4, 4, CV_64F);
    for(int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            b.at<double>(j, i) = a[i][j];
        }
    }

    return b;
}

inline glm::mat4 CvMatToGlm4x4(cv::Mat a) {
    glm::mat4 b;
    for(int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            b[i][j] = a.at<double>(j, i);
        }
    }

    return b;
}


void HelloArApplication::OnDrawFrame(bool depthColorVisualizationEnabled,
                                     bool useDepthForOcclusion) {
  // Render the scene.
  glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);

  if (ar_session_ == nullptr) return;

  ArSession_setCameraTextureName(ar_session_,
                                 background_renderer_.GetTextureId());

  // Update session to get current frame and render camera background.
  if (ArSession_update(ar_session_, ar_frame_) != AR_SUCCESS) {
    LOGE("HelloArApplication::OnDrawFrame ArSession_update error");
  }

  ArCamera* ar_camera;
  ArFrame_acquireCamera(ar_session_, ar_frame_, &ar_camera);

  glm::mat4 view_mat;
  glm::mat4 projection_mat;
  ArCamera_getViewMatrix(ar_session_, ar_camera, glm::value_ptr(view_mat));
  ArCamera_getProjectionMatrix(ar_session_, ar_camera,
                               0.1f, 100.f,
                               glm::value_ptr(projection_mat));

  background_renderer_.Draw(ar_session_, ar_frame_);

  ArTrackingState camera_tracking_state;
  ArCamera_getTrackingState(ar_session_, ar_camera, &camera_tracking_state);
  ArCamera_release(ar_camera);

  // If the camera isn't tracking don't bother rendering other objects.
  if (camera_tracking_state != AR_TRACKING_STATE_TRACKING) {
    return;
  }

    // Update and render point cloud.
    ArPointCloud* ar_point_cloud = nullptr;
    ArStatus point_cloud_status =
            ArFrame_acquirePointCloud(ar_session_, ar_frame_, &ar_point_cloud);
    if (point_cloud_status == AR_SUCCESS) {
        point_cloud_renderer_.Draw(projection_mat * view_mat, ar_session_, ar_point_cloud);
        ArPointCloud_release(ar_point_cloud);
    }

  cv::Mat extrinsics = View::oglExtrinsicsToCVExtrinsics(glm4x4ToCvMat(view_mat));
  std::future_status reconstruction_status;

    if (reconstructionFuture.valid()) {
        reconstruction_status = reconstructionFuture.wait_for(std::chrono::milliseconds(0));
    }
    else {
        reconstruction_status = std::future_status::ready;
    }

    // Decide weather to store the new image, load it if yes, and decide with which other images to perform comparision
  std::pair<float, float> relativePose;
  if (slideWindow.size() > 0) {
      relativePose = View::getRelativeRotationAndTranslation(extrinsics, slideWindow.getView(0).extrinsics);
  }
  if (slideWindow.size() == 0 || relativePose.first >= 0.2 || relativePose.second >= 0.1) {
      LOGI("Adding Image");
      const GLuint texId = background_renderer_.GetTextureId();
      glBindTexture(GL_TEXTURE_EXTERNAL_OES, texId);
      int lwidth, lheight;
      glGetTexLevelParameteriv(GL_TEXTURE_EXTERNAL_OES, 0, GL_TEXTURE_WIDTH, &lwidth);
      glGetTexLevelParameteriv(GL_TEXTURE_EXTERNAL_OES, 0, GL_TEXTURE_HEIGHT, &lheight);

      GLubyte* pixels = new GLubyte[lwidth * lheight * 4]; // 4 channels (RGBA)

      GLuint fbo;
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_EXTERNAL_OES, texId, 0);

      glReadPixels(0, 0, lwidth, lheight, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fbo);
      util::CheckGlError("Something went wrong when copying the image to CPU");

      cv::Mat image(lheight, lwidth, CV_8UC4, pixels);
      cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
      cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
      cv::resize(image, image, cv::Size(720, 1280));

      delete[] pixels;

      cv::Mat intrinsics = View::oglIntrinsicsToCVIntrinsics(glm4x4ToCvMat(projection_mat), image.size());
      View current(image, intrinsics, extrinsics);
      int currentIndex = slideWindow.add_image(current);

      std::set<int> exbaselines;
      const float ImageSamplingDensity = 0.2;
      for (int i = 1; i < slideWindow.size(); i++) {
          auto rel = View::getRelativeRotationAndTranslation(slideWindow.getView(-i).extrinsics, current.extrinsics);

          if (rel.first < 0.1) {
              int mbaseline = (int)(rel.second / ImageSamplingDensity);
              if (exbaselines.find(mbaseline) == exbaselines.end()) {
                  LOGI("Computation: %d, %d", currentIndex, currentIndex - i);
                  exbaselines.insert(mbaseline);
                  bufferedComputations.push_back(std::make_pair(currentIndex, currentIndex - i));
              }
          }
      }
  }

  // Render dense pointcloud, potentially updating if reconstruction not running
    int updates = 0;
    if (reconstruction_status == std::future_status::ready) {
        updates = updatedPointsForRender.size();
    }
    // reconstructorOutput and collectedScene are only accessed if updates > 0
    densePointRenderer_.draw(collectedScene, updatedPointsForRender, updates, projection_mat * view_mat);
    if (reconstruction_status == std::future_status::ready) {
        updatedPointsForRender.clear();
    }


    // Try to start a new computation if none is running
  if (bufferedComputations.size() > 0 && reconstruction_status == std::future_status::ready) {
      long lastDefinedImage = (long)slideWindow.getCurrentImageIndex() - (long)slideWindow.size() + 1;
      bool isComputationRunning = false;
      while (bufferedComputations.size() > 0) {
          currentComputation = bufferedComputations.back();
          bufferedComputations.pop_back();
          isComputationRunning = currentComputation.first >= lastDefinedImage && currentComputation.second >= lastDefinedImage;
          if (isComputationRunning) {
              break;
          }
          else {
              LOGI("Lost a computation cause image already erased");
          }
      }

      if (isComputationRunning) {
          View v1 = slideWindow.getViewByImageIndex(currentComputation.first);
          View v2 = slideWindow.getViewByImageIndex(currentComputation.second);

          reconstructionFuture = std::async(std::launch::async, [this, v1, v2] {
              auto start_time = std::chrono::high_resolution_clock::now();
              dbgidx++;
              LOGI("Starting computation %d, %d", currentComputation.first, currentComputation.second);

              reconstructorOutput.emplace_back();
              Reconstruct::compute3d(v1, v2, reconstructorOutput.back(), 1.0, 10, collectedScene.retrieveVoxelSidelength(3), 16 * 15);

              unfiltered_points += reconstructorOutput.back().size();
              updatedPointsForRender.push_back(reconstructorOutput.back());
              for (const ScenePoint &a: reconstructorOutput.back()) {
                  collectedScene.addPoint(ScenePoint(a.position, a.normal, 1));
              }

              auto it = reconstructorOutput.begin();
              while (unfiltered_points > 500000) {
                  LOGI("Running filter %d", unfiltered_points);
                  collectedScene.filterOutliers<1>(0, 200, *it);
                  collectedScene.filterOutliers<2>(1, 40, *it);
                  unfiltered_points -= it->size();
                  updatedPointsForRender.push_back(*it);

                  it++;
              }

              auto end_time = std::chrono::high_resolution_clock::now();
              auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time);
              LOGI("Ended %d", duration.count());
          });
      }
  }
}

 void HelloArApplication::ComputeSurface() {
    //collectedScene.filterConfidence(4);
    //collectedScene.filterOutliers(10, 200);
    collectedScene.filterOutliers<1>(0, 200);
    collectedScene.filterOutliers<2>(1, 40);
    collectedScene.export_xyz("/data/data/com.google.ar.core.examples.c.helloar/out.xyz");
    //PoissonSurfaceReconstruct<int, float, 3>::reconstructAndExport(collectedScene, "/data/data/com.google.ar.core.examples.c.helloar/out.ply");
    LOGI("Done");
}


    std::string matToStringExport(cv::Mat a) {
        std::string o;
        int row = 0;
        int col = 0;
        for (int k = 0; k < 16; k++) {
            double t = a.at<double>(row, col);
            o += std::to_string(t);
            if (k < 15) {
                o += ",";
            }
            row++;
            if (row == 4) {
                row = 0;
                col++;
            }
        }
        return o;
    }


void HelloArApplication::ConfigureSession() {

  ArConfig* ar_config = nullptr;
  ArConfig_create(ar_session_, &ar_config);
  ArConfig_setDepthMode(ar_session_, ar_config, AR_DEPTH_MODE_DISABLED);
  ArConfig_setInstantPlacementMode(ar_session_, ar_config,
                                   AR_INSTANT_PLACEMENT_MODE_DISABLED);
  ArConfig_setPlaneFindingMode(ar_session_, ar_config,
                             AR_PLANE_FINDING_MODE_DISABLED);

  CHECK(ar_config);
  CHECK(ArSession_configure(ar_session_, ar_config) == AR_SUCCESS);
  ArConfig_destroy(ar_config);
}

// This method returns a transformation matrix that when applied to screen space
// uvs makes them match correctly with the quad texture coords used to render
// the camera feed. It takes into account device orientation.
glm::mat3 HelloArApplication::GetTextureTransformMatrix(
    const ArSession* session, const ArFrame* frame) {
  float frameTransform[6];
  float uvTransform[9];
  // XY pairs of coordinates in NDC space that constitute the origin and points
  // along the two principal axes.
  const float ndcBasis[6] = {0, 0, 1, 0, 0, 1};
  ArFrame_transformCoordinates2d(
      session, frame, AR_COORDINATES_2D_OPENGL_NORMALIZED_DEVICE_COORDINATES, 3,
      ndcBasis, AR_COORDINATES_2D_TEXTURE_NORMALIZED, frameTransform);

  // Convert the transformed points into an affine transform and transpose it.
  float ndcOriginX = frameTransform[0];
  float ndcOriginY = frameTransform[1];
  uvTransform[0] = frameTransform[2] - ndcOriginX;
  uvTransform[1] = frameTransform[3] - ndcOriginY;
  uvTransform[2] = 0;
  uvTransform[3] = frameTransform[4] - ndcOriginX;
  uvTransform[4] = frameTransform[5] - ndcOriginY;
  uvTransform[5] = 0;
  uvTransform[6] = ndcOriginX;
  uvTransform[7] = ndcOriginY;
  uvTransform[8] = 1;

  return glm::make_mat3(uvTransform);
}
}  // namespace hello_ar

/*
void HelloArApplication::OnDrawFrame(bool depthColorVisualizationEnabled,
                                     bool useDepthForOcclusion) {
    // Render the scene.
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    if (ar_session_ == nullptr) return;

    ArSession_setCameraTextureName(ar_session_,
                                   background_renderer_.GetTextureId());

    // Update session to get current frame and render camera background.
    if (ArSession_update(ar_session_, ar_frame_) != AR_SUCCESS) {
        LOGE("HelloArApplication::OnDrawFrame ArSession_update error");
    }

    if (!dbgexport.is_open()) {
        dbgexport.open("/data/data/com.google.ar.core.examples.c.helloar/ARCoreData.json", std::ios_base::out);
        dbgexport << "{\n \"ARCoreData\": [";
    }

    ArCamera* ar_camera;
    ArFrame_acquireCamera(ar_session_, ar_frame_, &ar_camera);

    glm::mat4 view_mat;
    glm::mat4 projection_mat;
    ArCamera_getViewMatrix(ar_session_, ar_camera, glm::value_ptr(view_mat));
    ArCamera_getProjectionMatrix(ar_session_, ar_camera,
                                 0.1f, 100.f,
                                 glm::value_ptr(projection_mat));

    background_renderer_.Draw(ar_session_, ar_frame_);

    ArTrackingState camera_tracking_state;
    ArCamera_getTrackingState(ar_session_, ar_camera, &camera_tracking_state);
    ArCamera_release(ar_camera);

    // If the camera isn't tracking don't bother rendering other objects.
    if (camera_tracking_state != AR_TRACKING_STATE_TRACKING) {
        return;
    }

    cv::Mat extrinsics = glm4x4ToCvMat(view_mat);

    // Read the image from GPU (acquireCameraImage gives YUV format => useless unless writing conversion to RGB for 2 hours) ;(
    const GLuint texId = background_renderer_.GetTextureId();
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texId);
    int lwidth, lheight;
    glGetTexLevelParameteriv(GL_TEXTURE_EXTERNAL_OES, 0, GL_TEXTURE_WIDTH, &lwidth);
    glGetTexLevelParameteriv(GL_TEXTURE_EXTERNAL_OES, 0, GL_TEXTURE_HEIGHT, &lheight);

    GLubyte* pixels = new GLubyte[lwidth * lheight * 4]; // 4 channels (RGBA)

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_EXTERNAL_OES, texId, 0);

    glReadPixels(0, 0, lwidth, lheight, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    util::CheckGlError("Something went wrong when copying the image to CPU");

    oldimages.push_back(pixels);
    cv::Mat image(lheight, lwidth, CV_8UC4, pixels);
    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);

    cv::resize(image, image, cv::Size(720, 1280));

    cv::Mat intrinsics = glm4x4ToCvMat(projection_mat);

    auto imname = std::to_string(dbgidx);
    cv::imwrite("/data/data/com.google.ar.core.examples.c.helloar/" + imname + ".jpg", image);
    dbgexport << ",{\"name\": " << "\"" << imname << "\"" << ",\n"
        << "\"projection\": [" << matToStringExport(intrinsics) << "],\n"
        << "\"viewmatrix\": [" << matToStringExport(extrinsics) << "]\n}\n";

    dbgidx++;
}

void HelloArApplication::ComputeSurface() {
    dbgexport << "]\n}";
    dbgexport.close();
    exit(0);
}*/
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

namespace hello_ar {

HelloArApplication::HelloArApplication(AAssetManager* asset_manager)
    : asset_manager_(asset_manager) {}

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

    ArSession_setDisplayGeometry(ar_session_, display_rotation_, width_,
                                 height_);
  }

  const ArStatus status = ArSession_resume(ar_session_);
  CHECKANDTHROW(status == AR_SUCCESS, env, "Failed to resume AR session.");
}

void HelloArApplication::OnSurfaceCreated() {
  LOGI("OnSurfaceCreated()");

  background_renderer_.InitializeGlContent(asset_manager_);
  point_cloud_renderer_.InitializeGlContent(asset_manager_);
}

void HelloArApplication::OnDisplayGeometryChanged(int display_rotation,
                                                  int width, int height) {
  LOGI("OnSurfaceChanged(%d, %d)", width, height);
  glViewport(0, 0, width, height);
  display_rotation_ = display_rotation;
  width_ = width;
  height_ = height;
  if (ar_session_ != nullptr) {
    ArSession_setDisplayGeometry(ar_session_, display_rotation, width, height);
  }
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
                               /*near=*/0.1f, /*far=*/100.f,
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
    point_cloud_renderer_.Draw(projection_mat * view_mat, ar_session_,
                               ar_point_cloud);
    ArPointCloud_release(ar_point_cloud);
  }
}

void HelloArApplication::ConfigureSession() {

  ArConfig* ar_config = nullptr;
  ArConfig_create(ar_session_, &ar_config);
  ArConfig_setDepthMode(ar_session_, ar_config, AR_DEPTH_MODE_DISABLED);
  ArConfig_setInstantPlacementMode(ar_session_, ar_config,
                                   AR_INSTANT_PLACEMENT_MODE_DISABLED);

  CHECK(ar_config);
  CHECK(ArSession_configure(ar_session_, ar_config) == AR_SUCCESS);
  ArConfig_destroy(ar_config);
}

/*
void HelloArApplication::OnTouched(float x, float y) {
  if (ar_frame_ != nullptr && ar_session_ != nullptr) {
    ArHitResultList* hit_result_list = nullptr;
    ArHitResultList_create(ar_session_, &hit_result_list);
    CHECK(hit_result_list);
    if (is_instant_placement_enabled_) {
      ArFrame_hitTestInstantPlacement(ar_session_, ar_frame_, x, y,
                                      kApproximateDistanceMeters,
                                      hit_result_list);
    } else {
      ArFrame_hitTest(ar_session_, ar_frame_, x, y, hit_result_list);
    }

    int32_t hit_result_list_size = 0;
    ArHitResultList_getSize(ar_session_, hit_result_list,
                            &hit_result_list_size);

    // The hitTest method sorts the resulting list by distance from the camera,
    // increasing.  The first hit result will usually be the most relevant when
    // responding to user input.

    ArHitResult* ar_hit_result = nullptr;
    for (int32_t i = 0; i < hit_result_list_size; ++i) {
      ArHitResult* ar_hit = nullptr;
      ArHitResult_create(ar_session_, &ar_hit);
      ArHitResultList_getItem(ar_session_, hit_result_list, i, ar_hit);

      if (ar_hit == nullptr) {
        LOGE("HelloArApplication::OnTouched ArHitResultList_getItem error");
        return;
      }

      ArTrackable* ar_trackable = nullptr;
      ArHitResult_acquireTrackable(ar_session_, ar_hit, &ar_trackable);
      ArTrackableType ar_trackable_type = AR_TRACKABLE_NOT_VALID;
      ArTrackable_getType(ar_session_, ar_trackable, &ar_trackable_type);
      // Creates an anchor if a plane or an oriented point was hit.
      if (AR_TRACKABLE_PLANE == ar_trackable_type) {
        ArPose* hit_pose = nullptr;
        ArPose_create(ar_session_, nullptr, &hit_pose);
        ArHitResult_getHitPose(ar_session_, ar_hit, hit_pose);
        int32_t in_polygon = 0;
        ArPlane* ar_plane = ArAsPlane(ar_trackable);
        ArPlane_isPoseInPolygon(ar_session_, ar_plane, hit_pose, &in_polygon);

        // Use hit pose and camera pose to check if hittest is from the
        // back of the plane, if it is, no need to create the anchor.
        ArPose* camera_pose = nullptr;
        ArPose_create(ar_session_, nullptr, &camera_pose);
        ArCamera* ar_camera;
        ArFrame_acquireCamera(ar_session_, ar_frame_, &ar_camera);
        ArCamera_getPose(ar_session_, ar_camera, camera_pose);
        ArCamera_release(ar_camera);
        float normal_distance_to_plane = util::CalculateDistanceToPlane(
            *ar_session_, *hit_pose, *camera_pose);

        ArPose_destroy(hit_pose);
        ArPose_destroy(camera_pose);

        if (!in_polygon || normal_distance_to_plane < 0) {
          continue;
        }

        ar_hit_result = ar_hit;
        break;
      } else if (AR_TRACKABLE_POINT == ar_trackable_type) {
        ArPoint* ar_point = ArAsPoint(ar_trackable);
        ArPointOrientationMode mode;
        ArPoint_getOrientationMode(ar_session_, ar_point, &mode);
        if (AR_POINT_ORIENTATION_ESTIMATED_SURFACE_NORMAL == mode) {
          ar_hit_result = ar_hit;
          break;
        }
      } else if (AR_TRACKABLE_INSTANT_PLACEMENT_POINT == ar_trackable_type) {
        ar_hit_result = ar_hit;
      } else if (AR_TRACKABLE_DEPTH_POINT == ar_trackable_type) {
        // ArDepthPoints are only returned if ArConfig_setDepthMode() is called
        // with AR_DEPTH_MODE_AUTOMATIC.
        ar_hit_result = ar_hit;
      }
    }

    if (ar_hit_result) {
      // Note that the application is responsible for releasing the anchor
      // pointer after using it. Call ArAnchor_release(anchor) to release.
      ArAnchor* anchor = nullptr;
      if (ArHitResult_acquireNewAnchor(ar_session_, ar_hit_result, &anchor) !=
          AR_SUCCESS) {
        LOGE(
            "HelloArApplication::OnTouched ArHitResult_acquireNewAnchor error");
        return;
      }

      ArTrackingState tracking_state = AR_TRACKING_STATE_STOPPED;
      ArAnchor_getTrackingState(ar_session_, anchor, &tracking_state);
      if (tracking_state != AR_TRACKING_STATE_TRACKING) {
        ArAnchor_release(anchor);
        return;
      }

      if (anchors_.size() >= kMaxNumberOfAndroidsToRender) {
        ArAnchor_release(anchors_[0].anchor);
        ArTrackable_release(anchors_[0].trackable);
        anchors_.erase(anchors_.begin());
      }

      ArTrackable* ar_trackable = nullptr;
      ArHitResult_acquireTrackable(ar_session_, ar_hit_result, &ar_trackable);
      // Assign a color to the object for rendering based on the trackable type
      // this anchor attached to. For AR_TRACKABLE_POINT, it's blue color, and
      // for AR_TRACKABLE_PLANE, it's green color.
      ColoredAnchor colored_anchor;
      colored_anchor.anchor = anchor;
      colored_anchor.trackable = ar_trackable;

      UpdateAnchorColor(&colored_anchor);
      anchors_.push_back(colored_anchor);

      ArHitResult_destroy(ar_hit_result);
      ar_hit_result = nullptr;

      ArHitResultList_destroy(hit_result_list);
      hit_result_list = nullptr;
    }
  }
}
*/

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

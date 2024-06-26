# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)
project("CheapMeshStereo-SAMPLE-APP")

if(CMAKE-BUILD-TYPE STREQUAL "Release")
    message("Configuring Release build")
    add_compile_options(-O3)
    add_compile_options(-ffast-math)
else()
    message("Configuring Debug build")
    add_compile_options(-g)
endif()

# Import Opencv
add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
                    ${OPENCV_INCLUDE}/libs/${ANDROID_ABI}/libopencv_java4.so
                    INTERFACE_INCLUDE_DIRECTORIES ${OPENCV_INCLUDE}/jni/include)

# Import the ARCore (Google Play Services for AR) library.
add_library(arcore SHARED IMPORTED)
set_target_properties(arcore PROPERTIES IMPORTED_LOCATION
                      ${ARCORE_LIBPATH}/${ANDROID_ABI}/libarcore_sdk_c.so
                      INTERFACE_INCLUDE_DIRECTORIES ${ARCORE_INCLUDE}
)

# Import the glm header file from the NDK.
add_library( glm INTERFACE )
set_target_properties( glm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${GLM_INCLUDE})

# This is the main app library.
add_library(hello_ar_native SHARED
           src/main/cpp/background_renderer.cc
           src/main/cpp/hello_ar_application.cc
           src/main/cpp/jni_interface.cc
           src/main/cpp/obj_renderer.cc
           src/main/cpp/plane_renderer.cc
           src/main/cpp/point_cloud_renderer.cc
           src/main/cpp/dense_point_renderer.cc
           src/main/cpp/texture.cc
           src/main/cpp/util.cc)

#add_library(poisson INTERFACE)
#set_target_properties(poisson PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${POISSON-INCLUDE})
target_include_directories(hello_ar_native PRIVATE ${POISSON-INCLUDE})

#target_include_directories(hello_ar_native PUBLIC ${MOBILE3D-CORE-INCLUDE})
add_library(mob-core INTERFACE)
set_target_properties( mob-core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${MOBILE3D-CORE-INCLUDE})

target_include_directories(hello_ar_native PRIVATE
           src/main/cpp)
target_link_libraries(hello_ar_native
                      android
                      log
                      GLESv3
                      glm
                      arcore
                      lib_opencv
                      mob-core)

//
// Created by Admin on 29.10.2023.
//

#ifndef SAMPLEAPP_DENSE_POINT_RENDERER_H
#define SAMPLEAPP_DENSE_POINT_RENDERER_H


#include <GLES3/gl31.h>
#include "glm.h"
#include <Scene.h>


class dense_point_renderer {
    GLfloat squareVertices[12] = {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            0.5f,  0.5f, 0.0f,
            -0.5f,  0.5f, 0.0f
    };

    GLuint squareIndices[6] = {
            0, 1, 2,
            2, 3, 0
    };

    struct InstanceData {
        cv::Vec3f position;
        cv::Vec3f normal;
    };

    const char* vertexShaderSource = R"(#version 300 es

        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aInstancePos;
        layout(location = 2) in vec3 aInstanceNormal;

        uniform mat4 projectionview;
        uniform float scale;

        out vec4 color;

        void main() {
            // Approach copied from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

            vec3 norotn = vec3(0.0, 0.0, 1.0);
            vec3 n = normalize(aInstanceNormal);
            vec3 t = norotn + n;
            float s = dot(t, t);

            mat3 R;
            if (s < 0.001) {
                R = mat3(-1.0);
            }
            else {
                R = (2.0 * mat3(t * t.x, t * t.y, t * t.z) / s) - mat3(1.0);
            }

            vec4 modelPos = vec4(R * (0.02 * aPos) * scale + aInstancePos, 1.0);
            gl_Position = projectionview * modelPos;

            color = vec4(0.7, 0.7, 0.7, 1.0);

        }
    )";

    const char* fragmentShaderSource = R"(#version 300 es

        precision mediump float;

        out vec4 FragColor;

        in vec4 color;

        void main() {
            FragColor = color;
        }
    )";

    GLuint compileShader(GLenum type, const char* source);

    GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);

    GLuint VBO, VAO, EBO, InstanceBuffer;
    GLuint shaderProgram;
    GLuint mvp_uniform;
    GLuint scale_uniform;

    InstanceData *data = nullptr;
    unsigned int dataSize = 0;
    unsigned int actualDataSize = 0;
    std::unordered_map<cv::Vec3i, size_t, VecHash> voxelsToIndex;
    std::unordered_map<size_t, cv::Vec3i> indexToVoxel;
    static const int SceneMaxLevel = 3;
    float drawscale = 1.0;

public:
    void InitializeGLContent();

    void setScale(float scale);

    void draw(Scene<SceneMaxLevel> &scene, const std::list<std::vector<ScenePoint>> &updates, int use_updates, const glm::mat4& mvp_matrix);

    void reset();

    ~dense_point_renderer();
};


#endif //SAMPLEAPP_DENSE_POINT_RENDERER_H

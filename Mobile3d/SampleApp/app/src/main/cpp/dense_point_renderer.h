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
        glm::vec3 position;
        glm::vec3 normal;
    };

    /*
    const char* vertexShaderSource = R"(
        #version 300 es
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aInstancePos;
        layout(location = 2) in vec3 aInstanceNormal;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            vec4 modelPos = vec4(aPos + aInstancePos, 1.0);
            gl_Position = projection * view * modelPos;
        }
    )";
    */

    const char* vertexShaderSource = R"(#version 300 es

        in vec3 aPos;

        uniform mat4 projectionview;
        uniform vec3 position;
        uniform vec3 normal;

        void main() {
            vec3 norotn = vec3(0.0, 0.0, 1.0);
            mat3 R = mat3(
                normalize(normal), normalize(cross(norotn, normal)), normalize(norotn)
            );

            vec4 modelPos = vec4(R * (2.0 * 0.01 * aPos) + position, 1.0);
            gl_Position = projectionview * modelPos;
        }
    )";

    const char* fragmentShaderSource = R"(#version 300 es

        precision mediump float;

        out vec4 FragColor;

        void main() {
            FragColor = vec4(0, 1.0, 0, 0.3);
        }
    )";

    GLuint compileShader(GLenum type, const char* source);

    GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);

    GLuint VBO, VAO, EBO;
    GLuint shaderProgram;
    GLuint mvp_uniform, normal_uniform, position_uniform;

public:
    void InitializeGLContent();

    void draw(Scene &scene, const glm::mat4& mvp_matrix);
};


#endif //SAMPLEAPP_DENSE_POINT_RENDERER_H

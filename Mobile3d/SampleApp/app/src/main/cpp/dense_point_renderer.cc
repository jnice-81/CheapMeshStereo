//
// Created by Admin on 29.10.2023.
//

#include "dense_point_renderer.h"
#include "util.h"

GLuint dense_point_renderer::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, sizeof(infoLog), NULL, infoLog);
        LOGI("%s", infoLog);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint dense_point_renderer::linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check for linking errors
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, sizeof(infoLog), NULL, infoLog);
        LOGI("%s", infoLog);
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

void dense_point_renderer::InitializeGLContent() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    shaderProgram = linkProgram(vertexShader, fragmentShader);

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    mvp_uniform = glGetUniformLocation(shaderProgram, "projectionview");
    normal_uniform = glGetUniformLocation(shaderProgram, "normal");
    position_uniform = glGetUniformLocation(shaderProgram, "position");

    hello_ar::util::CheckGlError("Something went wrong during initialization of dense point renderer");
}

void dense_point_renderer::draw(Scene &scene, const glm::mat4& mvp_matrix) {
    CHECK(shaderProgram);

    glDisable(GL_CULL_FACE);
    glUseProgram(shaderProgram);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(squareIndices), squareIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glUniformMatrix4fv(mvp_uniform, 1, GL_FALSE, glm::value_ptr(mvp_matrix));

    /*
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (GLvoid*)offsetof(InstanceData, position));
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1); // Set the divisor for the instance attribute

    // Instance normal attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (GLvoid*)offsetof(InstanceData, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    */

    auto points = scene.getScenePoints();
    for (auto m : points) {
        cv::Vec3f pos = scene.voxelToPoint(m.first);
        glUniform3fv(position_uniform, 1, pos.val);
        glUniform3fv(normal_uniform, 1, m.second.normal.val);

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        /*
        glm::vec4 g;
        g[0] = pos[0];
        g[1] = pos[1];
        g[2] = pos[2];
        g[3] = 1;
        glm::vec4 proj = mvp_matrix * g;
        proj = proj / proj[3];
        LOGI("(%f, %f, %f)", proj[0], proj[1], proj[2]);
        */
    }

    hello_ar::util::CheckGlError("UPS");

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO);
    glEnable(GL_CULL_FACE);
}
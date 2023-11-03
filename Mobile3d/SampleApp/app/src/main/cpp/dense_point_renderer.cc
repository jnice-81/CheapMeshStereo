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
    glGenBuffers(1, &InstanceBuffer);
    glGenVertexArrays(1, &VAO);

    mvp_uniform = glGetUniformLocation(shaderProgram, "projectionview");

    hello_ar::util::CheckGlError("Something went wrong during initialization of dense point renderer");
}

void dense_point_renderer::draw(Scene &scene, const glm::mat4& mvp_matrix) {
    CHECK(shaderProgram);

    glDisable(GL_CULL_FACE);
    glUseProgram(shaderProgram);
    hello_ar::util::CheckGlError("UPS");

    glBindVertexArray(VAO);

    hello_ar::util::CheckGlError("fdf");
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    hello_ar::util::CheckGlError("asdflkujl");
    glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
    hello_ar::util::CheckGlError("Rezo");

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(squareIndices), squareIndices, GL_STATIC_DRAW);

    hello_ar::util::CheckGlError("Uhu");

    auto points = scene.getScenePoints();
    if (points.size() > 0) {
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
        glEnableVertexAttribArray(0);

        InstanceData *load = new InstanceData[points.size()];
        int instanceId = 0;
        for (auto m : points) {
            load[instanceId].position = scene.voxelToPoint(m.first);
            load[instanceId].normal = m.second.normal;
            instanceId++;
        }

        glBindBuffer(GL_ARRAY_BUFFER, InstanceBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceData) * points.size(), load, GL_STATIC_DRAW);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (GLvoid*)offsetof(InstanceData, position));
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(1, 1);

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (GLvoid*)offsetof(InstanceData, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribDivisor(2, 1);

        glUniformMatrix4fv(mvp_uniform, 1, GL_FALSE, glm::value_ptr(mvp_matrix));

        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, points.size());

        delete[] load;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);

    hello_ar::util::CheckGlError("UPS");
}
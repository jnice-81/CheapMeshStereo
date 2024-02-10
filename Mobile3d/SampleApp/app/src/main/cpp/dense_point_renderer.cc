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

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(squareIndices), squareIndices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    hello_ar::util::CheckGlError("Something went wrong during initialization of dense point renderer");
}

void dense_point_renderer::draw(Scene<SceneMaxLevel, bool> &scene, const std::list<std::vector<ScenePoint>> &updates, int use_updates, const glm::mat4& mvp_matrix) {
    CHECK(shaderProgram);

    glUseProgram(shaderProgram);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    if (use_updates != 0) {
        std::unordered_set<cv::Vec3i, VecHash> updatedVoxels;
        auto it = updates.begin();
        for(int i = 0; i < use_updates; i++) {
            for(auto it2 = it->begin(); it2 != it->end(); it2++) {
                updatedVoxels.insert(scene.retrieveVoxel(it2->position, SceneMaxLevel));
            }
            it++;
        }

        auto &surfacePoints = scene.surfacePoints;
        for (const auto &voxel : updatedVoxels) {

            auto it = surfacePoints.findVoxel<SceneMaxLevel>(scene.retrievePoint(voxel, SceneMaxLevel));
            bool erase = it.isEnd();

            auto g = voxelsToIndex.find(voxel);

            size_t index;
            if (g == voxelsToIndex.end()) {
                if (erase) {
                    continue;
                }
                index = actualDataSize;
                actualDataSize++;
                if (actualDataSize > dataSize) {
                    size_t newsize = dataSize * 2 + 100000;
                    InstanceData *newdata = new InstanceData[newsize];
                    std::copy(data, &data[dataSize], newdata);
                    dataSize = newsize;
                    if (data != nullptr) {
                        delete[] data;
                    }
                    data = newdata;
                }
                voxelsToIndex.insert(std::make_pair(voxel, index));
                indexToVoxel.insert(std::make_pair(index, voxel));
            }
            else {
                index = g->second;
            }

            if (erase) {
                actualDataSize--;
                int swapwith = actualDataSize;

                if (index != swapwith) {
                    data[index] = data[swapwith];

                    cv::Vec3i npos = indexToVoxel[swapwith];
                    voxelsToIndex.erase(g);
                    indexToVoxel.erase(swapwith);
                    indexToVoxel[index] = npos;
                    voxelsToIndex[npos] = index;
                }
                else {
                    voxelsToIndex.erase(g);
                    indexToVoxel.erase(index);
                }
            }
            else {
                const ScenePoint& current = it->second;
                data[index].position = current.position;
                data[index].normal = current.normal;
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, InstanceBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceData) * actualDataSize, data, GL_STATIC_DRAW);
    }

    glBindBuffer(GL_ARRAY_BUFFER, InstanceBuffer);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (GLvoid*)offsetof(InstanceData, position));
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (GLvoid*)offsetof(InstanceData, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glUniformMatrix4fv(mvp_uniform, 1, GL_FALSE, glm::value_ptr(mvp_matrix));

    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, actualDataSize);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    hello_ar::util::CheckGlError("Error when drawing dense points");
}

dense_point_renderer::~dense_point_renderer() {
    if (data != nullptr) {
        delete[] data;
    }
}
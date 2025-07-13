/*

   Copyright 2023 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#pragma once
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <filesystem>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

// KRR_NAMESPACE_BEGIN

namespace glu {
    class Texture2D {
    public:
    private:
        uint32_t m_handle;
        GLenum m_format;
        GLsizei m_width;
        GLsizei m_height;
        uint32_t m_numMipLevels;

        struct {
            unsigned int m_initialized : 1;
        };

        Texture2D(const Texture2D &) = delete;
        Texture2D &operator=(const Texture2D &) = delete;

    public:
        Texture2D();
        ~Texture2D();

        Texture2D(Texture2D &&b);
        Texture2D &operator=(Texture2D &&b);

        void initialize(GLenum format, GLsizei width, GLsizei height, uint32_t numMipLevels);
        void finalize();
        bool isInitialized() const {
            return m_initialized;
        }

        void transferImage(GLenum format, GLenum type, const void* data, uint32_t mipLevel) const;
        void transferCompressedImage(const void* data, GLsizei size, uint32_t mipLevel) const;

        GLuint getHandle() const {
            return m_handle;
        }
    };
}

// KRR_NAMESPACE_END

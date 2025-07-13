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

#include "gl_util.h"

#ifdef GLUPlatform_Windows_MSVC
#   include <Windows.h>
#   undef near
#   undef far
#   undef min
#   undef max
#endif

#include <fstream>
#include <algorithm>
#include <set>
#include <regex>



namespace glu {
    template <typename... Types>
    static void throwRuntimeError(bool expr, const char* fmt, const Types &... args) {
        if (expr)
            return;

        char str[2048];
        snprintf(str, sizeof(str), fmt, args...);
        throw std::runtime_error(str);
    }

    Texture2D::Texture2D() :
        m_handle(0),
        m_format(0),
        m_width(0), m_height(0), m_numMipLevels(1),
        m_initialized(false) {
    }

    Texture2D::~Texture2D() {
        if (m_initialized)
            finalize();
    }

    Texture2D::Texture2D(Texture2D &&b) {
        m_handle = b.m_handle;
        m_format = b.m_format;
        m_width = b.m_width;
        m_height = b.m_height;
        m_numMipLevels = b.m_numMipLevels;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    Texture2D &Texture2D::operator=(Texture2D &&b) {
        finalize();

        m_handle = b.m_handle;
        m_format = b.m_format;
        m_width = b.m_width;
        m_height = b.m_height;
        m_numMipLevels = b.m_numMipLevels;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }

    void Texture2D::initialize(GLenum format, GLsizei width, GLsizei height, uint32_t numMipLevels) {
        throwRuntimeError(!m_initialized, "Texture2D is already initialized.");

        m_format = format;
        m_width = width;
        m_height = height;
        m_numMipLevels = numMipLevels;

        glCreateTextures(GL_TEXTURE_2D, 1, &m_handle);
        glTextureStorage2D(m_handle, m_numMipLevels, m_format, m_width, m_height);
        glTextureParameteri(m_handle, GL_TEXTURE_BASE_LEVEL, 0);
        glTextureParameteri(m_handle, GL_TEXTURE_MAX_LEVEL, m_numMipLevels - 1);

        m_initialized = true;
    }

    void Texture2D::finalize() {
        if (!m_initialized)
            return;

        glDeleteTextures(1, &m_handle);
        m_handle = 0;

        m_width = 0;
        m_height = 0;
        m_numMipLevels = 0;

        m_initialized = false;
    }

    void Texture2D::transferImage(GLenum format, GLenum type, const void* data, uint32_t mipLevel) const {
        glTextureSubImage2D(
            m_handle, mipLevel,
            0, 0,
            std::max<GLsizei>(m_width >> mipLevel, 1u), std::max<GLsizei>(m_height >> mipLevel, 1u),
            format, type, data);
    }

    void Texture2D::transferCompressedImage(const void* data, GLsizei size, uint32_t mipLevel) const {
        glCompressedTextureSubImage2D(
            m_handle, mipLevel,
            0, 0,
            std::max<GLsizei>(m_width >> mipLevel, 1u), std::max<GLsizei>(m_height >> mipLevel, 1u),
            m_format, size, data);
    }
}



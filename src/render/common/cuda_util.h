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


// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define CUDAUPlatform_Windows
#   if defined(__MINGW32__) // Defined for both 32 bit/64 bit MinGW
#       define CUDAUPlatform_Windows_MinGW
#   elif defined(_MSC_VER)
#       define CUDAUPlatform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define CUDAU_CODE_COMPLETION
#       endif
#   endif
#elif defined(__linux__)
#   define CUDAUPlatform_Linux
#elif defined(__APPLE__)
#   define CUDAUPlatform_macOS
#elif defined(__OpenBSD__)
#   define CUDAUPlatform_OpenBSD
#endif

#if __cplusplus <= 199711L
#   if defined(CUDAUPlatform_Windows_MSVC)
#       pragma message("\"/Zc:__cplusplus\" compiler option to enable the updated __cplusplus definition is recommended.")
#   else
#       pragma message("Enabling the updated __cplusplus definition is recommended.")
#   endif
#endif


#include <stdexcept>
#include <sstream>


#if defined(__CUDACC_RTC__)
// Defining things corresponding to cstdint and cfloat is left to the user.
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;
#else
#include <cstdint>
#include <cfloat>
#if defined(CUDAUPlatform_Windows)
#   pragma warning(push)
#   pragma warning(disable:4819)
#endif
#include <cuda.h>
#if defined(CUDAUPlatform_Windows)
#   pragma warning(pop)
#endif
#endif

#if !defined(__CUDA_ARCH__)
#   include <cstdio>
#   include <cstdlib>

#   include <algorithm>
#   include <vector>
#   include <sstream>

// JP: CUDA/OpenGL連携機能が不要な場合はコンパイルオプションとして
//     CUDA_UTIL_DONT_USE_GL_INTEROPの定義を行う。
//     GL/gl3w.hは必要に応じて書き換える。
// EN: Define CUDA_UTIL_DONT_USE_GL_INTEROP as a compile option if CUDA/OpenGL interoperability
//     is not required.
//     Modify GL/gl3w.h as needed.
#   if !defined(CUDA_UTIL_DONT_USE_GL_INTEROP)
#       define CUDA_UTIL_USE_GL_INTEROP
#   endif
#   if defined(CUDA_UTIL_USE_GL_INTEROP)
#       include "GL/glew.h"
#       include "GLFW/glfw3.h"
#       include <cudaGL.h>
#   endif

#   undef min
#   undef max
#   undef near
#   undef far
#   undef RGB
#endif

#if __cplusplus >= 202002L
#   include <concepts>
#endif



#if defined(__CUDACC__)
#   define CUDA_SHARED_MEM __shared__
#   define CUDA_CONSTANT_MEM __constant__
#   define CUDA_DEVICE_MEM __device__
#   define CUDA_DEVICE_KERNEL extern "C" __global__
#   define CUDA_INLINE __forceinline__
#   define CUDA_DEVICE_FUNCTION __device__
#   define CUDA_COMMON_FUNCTION __host__ __device__
#else
#   define CUDA_SHARED_MEM
#   define CUDA_CONSTANT_MEM
#   define CUDA_DEVICE_MEM
#   define CUDA_DEVICE_KERNEL
#   define CUDA_INLINE inline
#   define CUDA_DEVICE_FUNCTION
#   define CUDA_COMMON_FUNCTION
#endif



#ifdef _DEBUG
#   define CUDAU_ENABLE_ASSERT
#endif

#if defined(CUDAU_ENABLE_ASSERT)
#   if defined(__CUDA_ARCH__)
#       define CUDAUAssert(expr, fmt, ...) \
do { \
    if (!(expr)) { \
        printf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        printf(fmt"\n", ##__VA_ARGS__); \
    } \
} \
while (0)
#   else
#       define CUDAUAssert(expr, fmt, ...) \
do { \
    if (!(expr)) { \
        cudau::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        cudau::devPrintf(fmt"\n", ##__VA_ARGS__); \
        abort(); \
    } \
} \
while (0)
#   endif
#else
#   define CUDAUAssert(expr, fmt, ...)
#endif

#define CUDAUAssert_ShouldNotBeCalled() CUDAUAssert(false, "Should not be called!")
#define CUDAUAssert_NotImplemented() CUDAUAssert(false, "Not implemented yet!")

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)


namespace cudau {
    enum class ArrayElementType {
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        Float16,
        Float32,
        BC1_UNorm,
        BC1_UNorm_sRGB,
        BC2_UNorm,
        BC2_UNorm_sRGB,
        BC3_UNorm,
        BC3_UNorm_sRGB,
        BC4_UNorm,
        BC4_SNorm,
        BC5_UNorm,
        BC5_SNorm,
        BC6H_UF16,
        BC6H_SF16,
        BC7_UNorm,
        BC7_UNorm_sRGB
    };

    enum class ArraySurface {
        Enable = 0,
        Disable,
    };

    enum class BufferMapFlag {
        Unmapped = 0,
        ReadWrite,
        ReadOnly,
        WriteOnlyDiscard
    };

    enum class ArrayTextureGather {
        Enable = 0,
        Disable,
    };

    inline bool isBCFormat(ArrayElementType elemType) {
        return (elemType == cudau::ArrayElementType::BC1_UNorm ||
                elemType == cudau::ArrayElementType::BC1_UNorm_sRGB ||
                elemType == cudau::ArrayElementType::BC2_UNorm ||
                elemType == cudau::ArrayElementType::BC2_UNorm_sRGB ||
                elemType == cudau::ArrayElementType::BC3_UNorm ||
                elemType == cudau::ArrayElementType::BC3_UNorm_sRGB ||
                elemType == cudau::ArrayElementType::BC4_UNorm ||
                elemType == cudau::ArrayElementType::BC4_SNorm ||
                elemType == cudau::ArrayElementType::BC5_UNorm ||
                elemType == cudau::ArrayElementType::BC5_SNorm ||
                elemType == cudau::ArrayElementType::BC6H_UF16 ||
                elemType == cudau::ArrayElementType::BC6H_SF16 ||
                elemType == cudau::ArrayElementType::BC7_UNorm ||
                elemType == cudau::ArrayElementType::BC7_UNorm_sRGB);
    }

#   if defined(CUDA_UTIL_USE_GL_INTEROP)
    void getArrayElementFormat(GLenum internalFormat, ArrayElementType* elemType, uint32_t* numChannels);
#   endif

    class Array {
        CUcontext m_cuContext;

        size_t m_width;
        size_t m_height;
        size_t m_depth;
        uint32_t m_numMipmapLevels;
        uint32_t m_stride;
        ArrayElementType m_elemType;
        uint32_t m_numChannels;

        union {
            CUarray m_array;
            CUmipmappedArray m_mipmappedArray;
        };
        void** m_mappedPointers;
        CUarray* m_mipmapArrays;
        BufferMapFlag* m_mapFlags;
        CUsurfObject* m_surfObjs;

        uint32_t m_GLTexID;
        CUgraphicsResource m_cudaGfxResource;

        struct {
            unsigned int m_surfaceLoadStore : 1;
            unsigned int m_useTextureGather : 1;
            unsigned int m_cubemap : 1;
            unsigned int m_layered : 1;
            unsigned int m_initialized : 1;
        };

        Array(const Array &) = delete;
        Array &operator=(const Array &) = delete;

        void initialize(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            size_t width, size_t height, size_t depth, uint32_t numMipmapLevels,
            bool writable, bool useTextureGather, bool cubemap, bool layered, uint32_t glTexID);

        void computeDimensionsOfLevel(uint32_t mipmapLevel, size_t* width, size_t* height) const {
            *width = std::max<size_t>(1, m_width >> mipmapLevel);
            *height = std::max<size_t>(1, m_height >> mipmapLevel);
            if (isBCFormat(m_elemType)) {
                *width = (*width + 3) / 4;
                *height = (*height + 3) / 4;
            }
        }

    public:
        Array();
        ~Array();

        Array(Array &&b);
        Array &operator=(Array &&b);

        void initialize1D(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            ArraySurface surfaceLoadStore,
            size_t length, uint32_t numMipmapLevels) {
            initialize(
                context, elemType, numChannels, length, 0, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable, false, false, false, 0);
        }
        void initialize2D(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            ArraySurface surfaceLoadStore, ArrayTextureGather useTextureGather,
            size_t width, size_t height, uint32_t numMipmapLevels) {
            initialize(
                context, elemType, numChannels, width, height, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable,
                useTextureGather == ArrayTextureGather::Enable,
                false, false, 0);
        }
        void initialize3D(
            CUcontext context, ArrayElementType elemType, uint32_t numChannels,
            ArraySurface surfaceLoadStore,
            size_t width, size_t height, size_t depth, uint32_t numMipmapLevels) {
            initialize(
                context, elemType, numChannels, width, height, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable, false, false, false, 0);
        }
        void initializeFromGLTexture2D(
            CUcontext context, uint32_t glTexID,
            ArraySurface surfaceLoadStore, ArrayTextureGather useTextureGather) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            GLint width, height;
            GLint numMipmapLevels;
            GLint format;
            glGetTextureLevelParameteriv(glTexID, 0, GL_TEXTURE_WIDTH, &width);
            glGetTextureLevelParameteriv(glTexID, 0, GL_TEXTURE_HEIGHT, &height);
            glGetTextureLevelParameteriv(glTexID, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
            glGetTextureParameteriv(glTexID, GL_TEXTURE_VIEW_NUM_LEVELS, &numMipmapLevels);
            numMipmapLevels = std::max(numMipmapLevels, 1);
            ArrayElementType elemType;
            uint32_t numChannels;
            getArrayElementFormat((GLenum)format, &elemType, &numChannels);
            initialize(
                context, elemType, numChannels, width, height, 0, numMipmapLevels,
                surfaceLoadStore == ArraySurface::Enable,
                useTextureGather == ArrayTextureGather::Enable,
                false, false, glTexID);
#else
            (void)context;
            (void)glTexID;
            (void)surfaceLoadStore;
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
        void finalize();

        void resize(size_t length, CUstream stream = 0);
        void resize(size_t width, size_t height, CUstream stream = 0);
        void resize(size_t width, size_t height, size_t depth, CUstream stream = 0);

        CUarray getCUarray(uint32_t mipmapLevel) const {
            if (m_GLTexID) {
                if (m_mipmapArrays[mipmapLevel] == nullptr)
                    throw std::runtime_error("This mip level of this interop array is not mapped.");
                return m_mipmapArrays[mipmapLevel];
            }
            else {
                if (m_numMipmapLevels > 1)
                    return m_mipmapArrays[mipmapLevel];
                else
                    return m_array;
            }
        }
        CUmipmappedArray getCUmipmappedArray() const {
            return m_mipmappedArray;
        }

        size_t getWidth() const {
            return m_width;
        }
        size_t getHeight() const {
            return m_height;
        }
        size_t getDepth() const {
            return m_depth;
        }
        uint32_t getNumMipmapLevels() const {
            return m_numMipmapLevels;
        }
        bool isBCTexture() const {
            return isBCFormat(m_elemType);
        }
        bool isInitialized() const {
            return m_initialized;
        }

        void beginCUDAAccess(CUstream stream, uint32_t mipmapLevel);
        void endCUDAAccess(CUstream stream, uint32_t mipmapLevel);

        void* map(
            uint32_t mipmapLevel = 0,
            CUstream stream = 0,
            BufferMapFlag flag = BufferMapFlag::ReadWrite);
        template <typename T>
        T* map(
            uint32_t mipmapLevel = 0,
            CUstream stream = 0,
            BufferMapFlag flag = BufferMapFlag::ReadWrite) {
            return reinterpret_cast<T*>(map(mipmapLevel, stream, flag));
        }
        void unmap(uint32_t mipmapLevel = 0, CUstream stream = 0);
        template <typename T>
        void write(
            const T* srcValues, size_t numValues, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            size_t depth = std::max<size_t>(1, m_depth);

            size_t bw;
            size_t bh;
            computeDimensionsOfLevel(mipmapLevel, &bw, &bh);
            size_t sizePerRow = bw * m_stride;
            size_t size = depth * bh * sizePerRow;
            if (sizeof(T) * numValues > size)
                throw std::runtime_error("Too large transfer.");
            size_t writeHeight = (sizeof(T) * numValues) / sizePerRow;

            CUDA_MEMCPY3D params = {};
            params.WidthInBytes = sizePerRow;
            params.Height = bh;
            params.Depth = depth;

            params.srcMemoryType = CU_MEMORYTYPE_HOST;
            params.srcHost = srcValues;
            params.srcPitch = sizePerRow;
            params.srcHeight = writeHeight;
            params.srcXInBytes = 0;
            params.srcY = 0;
            params.srcZ = 0;
            // srcArray, srcDevice, srcLOD are not used in this case.

            params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            params.dstArray = (m_numMipmapLevels > 1 || m_GLTexID != 0) ? m_mipmapArrays[mipmapLevel] : m_array;
            params.dstXInBytes = 0;
            params.dstY = 0;
            params.dstZ = 0;
            // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

            CUDADRV_CHECK(cuMemcpy3DAsync(&params, stream));
        }
        template <typename T>
        void write(const std::vector<T> &values, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            write(values.data(), values.size(), mipmapLevel, stream);
        }
        template <typename T>
        void read(T* dstValues, size_t numValues, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            size_t depth = std::max<size_t>(1, m_depth);

            size_t bw;
            size_t bh;
            computeDimensionsOfLevel(mipmapLevel, &bw, &bh);
            size_t sizePerRow = bw * m_stride;
            size_t size = depth * bh * sizePerRow;
            if (sizeof(T) * numValues > size)
                throw std::runtime_error("Too large transfer.");
            size_t readHeight = (sizeof(T) * numValues) / sizePerRow;

            CUDA_MEMCPY3D params = {};
            params.WidthInBytes = sizePerRow;
            params.Height = bh;
            params.Depth = depth;

            params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            params.srcArray = (m_numMipmapLevels > 1 || m_GLTexID != 0) ? m_mipmapArrays[mipmapLevel] : m_array;
            params.srcXInBytes = 0;
            params.srcY = 0;
            params.srcZ = 0;
            // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

            params.dstMemoryType = CU_MEMORYTYPE_HOST;
            params.dstHost = dstValues;
            params.dstPitch = sizePerRow;
            params.dstHeight = readHeight;
            params.dstXInBytes = 0;
            params.dstY = 0;
            params.dstZ = 0;
            // dstArray, dstDevice, dstLOD are not used in this case.

            CUDADRV_CHECK(cuMemcpy3DAsync(&params, stream));
        }
        template <typename T>
        void read(std::vector<T> &values, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            read(values.data(), values.size(), mipmapLevel, stream);
        }
        template <typename T>
        void fill(const T &value, uint32_t mipmapLevel = 0, CUstream stream = 0) const {
            size_t bw;
            size_t bh;
            computeDimensionsOfLevel(mipmapLevel, &bw, &bh);
            size_t depth = std::max<size_t>(1, m_depth);
            size_t sizePerRow = bw * m_stride;
            size_t size = depth * bh * sizePerRow;
            size_t numValues = size / sizeof(T);
            std::vector<T> values(value, numValues);
            write(values, mipmapLevel, stream);
        }

        CUDA_RESOURCE_VIEW_DESC getResourceViewDesc() const;

        CUsurfObject getSurfaceObject(uint32_t mipmapLevel) const {
            return m_surfObjs[mipmapLevel];
        }
        [[nodiscard]]
        CUsurfObject createGLSurfaceObject(uint32_t mipmapLevel) const {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            if (m_GLTexID == 0)
                throw std::runtime_error("This is not an array created from OpenGL object.");
            if (m_mipmapArrays[mipmapLevel] == nullptr)
                throw std::runtime_error("Use beginCUDAAccess()/endCUDAAccess().");

            CUsurfObject ret;
            CUDA_RESOURCE_DESC resDesc = {};
            resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            resDesc.res.array.hArray = m_mipmapArrays[mipmapLevel];
            CUDADRV_CHECK(cuSurfObjectCreate(&ret, &resDesc));
            return ret;
#else
            (void)mipmapLevel;
            throw std::runtime_error(
                "Disable \"CUDA_UTIL_DONT_USE_GL_INTEROP\" if you use CUDA/OpenGL interoperability.");
#endif
        }
    };


    // MIP-level 0 only
    template <uint32_t NumBuffers>
    class InteropSurfaceObjectHolder {
        Array* m_arrays[NumBuffers];
        CUsurfObject m_surfObjs[NumBuffers];
        uint32_t m_numArrays = 0;
        uint32_t m_arrayIndex = 0;
        uint32_t m_bufferIndex = 0;

    public:
        template <uint32_t numArrays>
        void initialize(Array* const (&arrays)[numArrays]) {
            for (uint32_t i = 0; i < NumBuffers; ++i)
                m_arrays[i] = arrays[i % numArrays];
            m_numArrays = numArrays;
            m_arrayIndex = 0;
            m_bufferIndex = 0;
            for (uint32_t i = 0; i < NumBuffers; ++i)
                m_surfObjs[i] = 0;
        }
        void finalize() {
            for (uint32_t i = 0; i < NumBuffers; ++i) {
                CUDADRV_CHECK(cuSurfObjectDestroy(m_surfObjs[i]));
                m_surfObjs[i] = 0;
            }
            m_bufferIndex = 0;
            m_arrayIndex = 0;
        }

        void beginCUDAAccess(CUstream stream) {
            m_arrays[m_arrayIndex]->beginCUDAAccess(stream, 0);
        }
        void endCUDAAccess(CUstream stream, bool endFrame) {
            m_arrays[m_arrayIndex]->endCUDAAccess(stream, 0);
            if (endFrame) {
                m_arrayIndex = (m_arrayIndex + 1) % m_numArrays;
                m_bufferIndex = (m_bufferIndex + 1) % NumBuffers;
            }
        }
        CUsurfObject getNext() {
            CUsurfObject &curSurfObj = m_surfObjs[m_bufferIndex];
            if (curSurfObj)
                CUDADRV_CHECK(cuSurfObjectDestroy(curSurfObj));
            curSurfObj = m_arrays[m_arrayIndex]->createGLSurfaceObject(0);
            return curSurfObj;
        }
    };
}
CMAKE_MINIMUM_REQUIRED ( VERSION 3.17 )

PROJECT (KRR-NPM
	DESCRIPTION "A simple codebase for ray-traced rendering techniques."
	LANGUAGES C CXX CUDA
)
SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(KRR_EXP_PROJECT_ROOT ${CMAKE_SOURCE_DIR})

###############################################################################
# Build type and C++ compiler setup
###############################################################################
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_DEBUG_POSTFIX "")

# the CONFIG inherently comes from CMAKE_BUILD_TYPE
add_compile_definitions ("$<$<CONFIG:Debug>:KRR_DEBUG_BUILD>")

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

set(CUDA_VERBOSE_BUILD ON)

if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

if (NOT CUDA_INCLUDE_DIRS)
	message(STATUS "Setting CUDA include directory to: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# auto detect compute arch, from https://stackoverflow.com/a/69353718
if (NOT DEFINED ENV{CUDA_ARCH})
	include(FindCUDA/select_compute_arch)
	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
	string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
	list(REMOVE_ITEM CUDA_ARCH_LIST "86+PTX")
	list(SORT CUDA_ARCH_LIST)
	message(STATUS "Automatically detected GPU architecture: ${CUDA_ARCH_LIST}")
	list(POP_BACK CUDA_ARCH_LIST CMAKE_CUDA_ARCHITECTURES)
	string(APPEND CMAKE_CUDA_ARCHITECTURES "-virtual")
	message(STATUS "Adopted GPU architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  	message(STATUS "Specified GPU architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

###############################################################################
# other dependencies
###############################################################################
find_package(OpenGL REQUIRED)

set(
    CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})
find_package(OptiX74)
find_package(OptiX73)

if (OptiX74_FOUND)
    set(OptiX_INCLUDE "${OPTIX74_INCLUDE_DIR}")
elseif(OptiX73_FOUND)
	set(OptiX_INCLUDE "${OPTIX73_INCLUDE_DIR}")
else()
	message(FATAL_ERROR "OptiX SDK 7.4 not found.")
endif()

ADD_SUBDIRECTORY (src)


SET (CMAKE_POLICY_DEFAULT_CMP0077 NEW)
STRING (REGEX REPLACE "/W[3|4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# glew
SET(glew-cmake_BUILD_SHARED OFF)
ADD_SUBDIRECTORY(glew EXCLUDE_FROM_ALL)
INCLUDE_DIRECTORIES(glew/include)
SET(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glew/include PARENT_SCOPE)

# vulkan memory allocator
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE vma)

# glfw3 (lib: glfw)
ADD_SUBDIRECTORY(glfw EXCLUDE_FROM_ALL)
INCLUDE_DIRECTORIES(glfw/include)
SET(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include PARENT_SCOPE)

# imgui	(lib: imgui)
ADD_SUBDIRECTORY(imgui)
SET(IMGUI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/imgui PARENT_SCOPE)

# Assimp
SET(BUILD_SHARED_LIBS OFF)
SET(ASSIMP_BUILD_TESTS OFF)
SET(ASSIMP_NO_EXPORT ON)
SET(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
# specify enabled importers for speed
SET(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF)
SET(ASSIMP_BUILD_OBJ_IMPORTER ON)
SET(ASSIMP_BUILD_FBX_IMPORTER ON)
SET(ASSIMP_BUILD_GLTF_IMPORTER ON)
ADD_SUBDIRECTORY(assimp EXCLUDE_FROM_ALL)
INCLUDE_DIRECTORIES(assimp/include)
SET(ASSIMP_INCLUDE_DIRS 
	${CMAKE_CURRENT_SOURCE_DIR}/assimp/include 
	${CMAKE_CURRENT_BINARY_DIR}/assimp/include	
	PARENT_SCOPE)

# for tinyexr, currently used the zlib contained in assimp, according to https://github.com/syoyo/tinyexr
# could switch to miniz if anything happens.
SET(IMAGEIO_INCLUDE_DIRS 
	${CMAKE_CURRENT_SOURCE_DIR}/image 
	${CMAKE_CURRENT_SOURCE_DIR}/assimp/contrib/zlib
	${CMAKE_CURRENT_BINARY_DIR}/assimp/contrib/zlib
	PARENT_SCOPE)
INCLUDE_DIRECTORIES(${IMAGEIO_INCLUDE_DIRS})

# json
SET(JSON_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/json
	PARENT_SCOPE)
INCLUDE_DIRECTORIES(${JSON_INCLUDE_DIRS})

# pybind11
IF(KRR_ENABLE_PYTHON)
	ADD_SUBDIRECTORY(pybind11)
ENDIF()

# pbrt parser
SET(pbrtParser_ENABLE_UNITTESTS OFF)
ADD_SUBDIRECTORY (pbrtparser EXCLUDE_FROM_ALL)
SET( PBRTPARSER_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/pbrtparser/pbrtParser/include
	PARENT_SCOPE)
INCLUDE_DIRECTORIES(PBRTPARSER_INCLUDE_DIRS)

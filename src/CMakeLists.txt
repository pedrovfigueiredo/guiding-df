set (KRR_BUILD_STARLIGHT ON)
add_subdirectory(lib)

# just a temporary workaround for tcnn dont pass definitions to custom cuda 
add_definitions(-DTCNN_MIN_GPU_ARCH=$ENV{CUDA_ARCH} -DTCNN_SHAMPOO)

add_subdirectory(ext)

set ( KRR_GUIDED_PT
	render/guided/parameters.h
	render/guided/guideditem.h
	render/guided/workqueue.h
	render/guided/guided.h
	render/guided/train.h
	render/guided/integrator.h
	render/guided/backend.h
	render/guided/AR.h
	render/guided/learnedDistribution.h
	render/guided/nrc.h
	render/common/gl_util.h
	render/common/cuda_util.h
	render/guided/integrator.cpp
	render/guided/backend.cpp
	render/guided/train.cu
	render/guided/learnedDistribution.cpp
	render/guided/nrc.cpp
	render/common/gl_util.cpp
	render/common/cuda_util.cpp
)

set_source_files_properties (
	render/guided/integrator.cpp
	render/guided/backend.cpp
	render/guided/learnedDistribution.cpp
	render/guided/nrc.cpp
	render/common/cuda_util.cpp
	PROPERTIES LANGUAGE CUDA
)

set ( KRR_EXP_INCLUDE_ALL
	${KRR_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${KRR_MISC_INCLUDE_ALL}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/ext/tcnn/include
	${CMAKE_CURRENT_SOURCE_DIR}/ext/tcnn/dependencies
)

message("KRR INCLUDE DIRS: ${KRR_EXP_INCLUDE_ALL}")

###############################################################################
# automatically creating definitions of structure of arrays (soa)
###############################################################################

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/render/guided/guideditem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/guided/guideditem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/render/guided/guideditem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/guided/guideditem.soa)

set (KRR_EXP_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/render/guided/guideditem_soa.h
)

add_custom_target (krr_exp_soa_generated DEPENDS ${KRR_EXP_SOA_GENERATED})

###############################################################################
# generating PTX code from optix shader routines
###############################################################################
include_directories (${KRR_EXP_INCLUDE_ALL})
include (${KRR_EXP_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
EXP_CUDA_COMPILE_EMBED(GUIDED_PTX render/guided/device.cu krr-guided krr_exp_soa_generated ${KRR_EXP_INCLUDE_ALL})
add_library(KRR_EXP_PTX STATIC
	# peseudo target for SOA generation priority
	${GUIDED_PTX}
)
add_dependencies(KRR_EXP_PTX krr_exp_soa_generated)

set ( KRR_EXP_LIBS_ALL
	OpenGL::GL
	libglew_static
	glfw
	krr_cuda_cfg
	krr_cuda_warning
	tiny-cuda-nn
	KRR_EXP_PTX
)

add_library (krr_exp_lib
	${KRR_GUIDED_PT}
	# SOA
	${KRR_EXP_SOA_GENERATED}
	# PTX 
	${GUIDED_PTX}
)
add_dependencies(krr_exp_lib krr_exp_soa_generated)
#set_property(TARGET krr_exp_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET krr_exp_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_link_libraries (krr_exp_lib PUBLIC ${KRR_EXP_LIBS_ALL})
target_include_directories(krr_exp_lib PUBLIC  ${KRR_EXP_INCLUDE_ALL})

add_executable( testbed main/main.cpp)
target_link_libraries (testbed PUBLIC ${KRR_EXP_LIBS_ALL})
target_link_libraries (testbed PUBLIC krr_exp_lib
	-WHOLEARCHIVE:$<TARGET_FILE:krr_exp_lib>)
target_link_libraries (testbed PUBLIC krr_lib
	-WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)

if (TARGET pykrr)
	message("Linking EXP library to python binding")
	TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_exp_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_exp_lib>)
endif()
	
# Looks for the environment variable:
# OPTIX73_PATH

# Sets the variables :
# OPTIX73_INCLUDE_DIR

# OptiX73_FOUND

set(OPTIX73_PATH $ENV{OPTIX74_PATH})

if ("${OPTIX73_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX73_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0")
  else()
    # Adjust this if the OptiX SDK 7.4.0 installation is in a different location.
    set(OPTIX73_PATH "~/NVIDIA-OptiX-SDK-7.3.0-linux64")
  endif()
endif()

find_path(OPTIX73_INCLUDE_DIR optix_7_host.h ${OPTIX73_PATH}/include)

# message("OPTIX74_INCLUDE_DIR = " "${OPTIX74_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX73 DEFAULT_MSG OPTIX73_INCLUDE_DIR)

mark_as_advanced(OPTIX73_INCLUDE_DIR)

# message("OptiX74_FOUND = " "${OptiX74_FOUND}")

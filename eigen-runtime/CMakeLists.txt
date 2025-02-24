# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
# Get Eigen.
find_package(Eigen3 REQUIRED NO_MODULE)

# Get OpenMP.
find_package(OpenMP REQUIRED)

set(EIGEN_RT_NAME "eigen-runtime")

# Get Eigen include dir.
get_target_property(EIGEN_INCLUDE_DIR
  Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

# Source for the runtime library.
set(runtime_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/dynamicEigen.cpp"
)

# Create runtime library.
add_llvm_library(${EIGEN_RT_NAME} SHARED "${runtime_srcs}")
target_include_directories(${EIGEN_RT_NAME} PRIVATE "${EIGEN_INCLUDE_DIR}")
target_include_directories(${EIGEN_RT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${EIGEN_RT_NAME} PUBLIC Eigen3::Eigen)
target_link_libraries(${EIGEN_RT_NAME} PUBLIC OpenMP::OpenMP_CXX)

# Add specialization flags.
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc.*")
  target_compile_options(${EIGEN_RT_NAME} PUBLIC -mcpu=native -mtune=native)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_compile_options(${EIGEN_RT_NAME} PUBLIC -march=native -mtune=native)
else()
  message(WARNING "Unknown specialization platform.")
endif()

# Add fast math flags on all platforms.
target_compile_options(${EIGEN_RT_NAME} PUBLIC -ffp-contract=fast -ffast-math)

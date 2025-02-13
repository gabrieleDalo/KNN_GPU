cmake_minimum_required(VERSION 3.22)
project(Finalterm-Assignment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specifica il percorso del CUDA Toolkit
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g --extended-lambda --forward-unknown-to-host-compiler")

# Aggiungi ottimizzazioni solo in modalità Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CMAKE_TOOLCHAIN_FILE "C:/Users/gabri/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")

# Trova la directory degli header di mlpack
find_path(MLPACK_INCLUDE_DIRS "mlpack.hpp" 
    PATHS "C:/Users/gabri/vcpkg/installed/x64-windows/include"
)

# Aggiungi include e link directory
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
include_directories("${MLPACK_INCLUDE_DIRS}")

add_executable(Finalterm-Assignment
        loadFiles.h
        loadFiles.cpp
        utils.cpp
        utils.h
        mainCUDA.cu
)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

target_link_libraries(Finalterm-Assignment PRIVATE ${ARMADILLO_LIBRARIES} BLAS::BLAS LAPACK::LAPACK)
set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:LIBCMT")
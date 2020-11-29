cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
# FIND_PACKAGE(CUDA 10.0)
include(CTest)
include_directories(./include)
include_directories(./external/include)
add_library(dgnn STATIC
include/data.h
include/dbg.h
include/spmm.h
include/util.h
include/dense.h
include/args.hxx
src/data.cu
src/util.cu
  )
  
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF() 
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(dgnn PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( dgnn
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp ${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70 -O2 --use_fast_math -Xptxas \"-v -dlcm=ca\" ")


if(BUILD_TESTING)
  #add_executable(test_bwd.out profile/test_bwd.cu)
  add_executable(fig7.out Figure7/main.cu)
  add_executable(fig8.out Figure8/main.cu)
  add_executable(fig9.out Figure9/main.cu)
  add_executable(fig10a.out Figure10/main_a.cu)
  add_executable(fig10b.out Figure10/main_b.cu)
  add_executable(fig11.out Figure11/main.cu)

  #set_target_properties(test_bwd.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #target_link_libraries(test_bwd.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

  set_target_properties(fig7.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fig7.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

  set_target_properties(fig8.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fig8.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

  set_target_properties(fig9.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fig9.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

  set_target_properties(fig10a.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fig10a.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

  set_target_properties(fig10b.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fig10b.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

  set_target_properties(fig11.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(fig11.out PRIVATE dgnn cublas curand OpenMP::OpenMP_CXX)

endif()




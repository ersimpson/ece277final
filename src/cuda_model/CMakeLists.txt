set(BLOCK_NAME mnist_cpp_model)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
	${CUDA_TOOLKIT_INCLUDE}
	${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
  ${TORCH_API_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIR}
)



file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})
cuda_add_library(${BLOCK_NAME} MODULE
	mnist_model.cpp
	mnist_cu_model.cu
OPTIONS --cudart=shared -DWITH_OPENGL -DOPTIMIZATION_STEP=0x0 -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
DEBUG -Xptxas -v -g -G 
)

set_target_properties(${BLOCK_NAME} PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX ".pyd"
)

target_link_libraries(${BLOCK_NAME}  ${PYTHON_LIBRARIES})
target_link_libraries( 
${BLOCK_NAME} 

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
${TORCH_DEBUG_LIBRARIES}
)

target_link_libraries( 
${BLOCK_NAME} 
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
${TORCH_RELEASE_LIBRARIES}
)

add_custom_command(TARGET ${BLOCK_NAME} POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/freeglut.dll"  
$<TARGET_FILE_DIR:${BLOCK_NAME}>)     

add_custom_command(TARGET ${BLOCK_NAME}  POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/glew64.dll"  
$<TARGET_FILE_DIR:${BLOCK_NAME}>) 

file(GLOB TORCH_RELEASE_DLLS "${TORCH_RELEASE_LIBPATH}/*.dll")
file(GLOB TORCH_DEBUG_DLLS "${TORCH_DEBUG_LIBPATH}/*.dll")

add_custom_command(TARGET ${BLOCK_NAME} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/c10.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/c10_cuda.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/torch.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/torch_cpu.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/torch_cuda.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/torch_cuda_cpp.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/torch_cuda_cu.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/torch_global_deps.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/fbgemm.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/asmjit.dll"
"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/uv.dll"
"${TORCH_TOP_DIR}/lib/Common/libiomp5md.dll"
"${TORCH_TOP_DIR}/lib/Common/nvToolsExt64_1.dll"
"${TORCH_TOP_DIR}/lib/Common/cusparse64_11.dll"
"${TORCH_TOP_DIR}/lib/Common/cusolver64_11.dll"
"${TORCH_TOP_DIR}/lib/Common/cudnn64_8.dll"
#"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/curand64_101.dll"
#"${TORCH_TOP_DIR}/lib/\$\(Configuration\)/cufftw64_101.dll"
"${TORCH_TOP_DIR}/lib/Common/cublas64_11.dll"
"${TORCH_TOP_DIR}/lib/Common/cublasLt64_11.dll"
$<TARGET_FILE_DIR:${BLOCK_NAME}>)

# Collect MyLoopPass src.
set(srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/source/DataAnalysis.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/source/Matcher.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/source/ReplacerPass.cpp"
)

# Create the KernelFaRer library.
llvm_add_library(KernelFaRer MODULE ${srcs})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_link_libraries(KernelFaRer "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

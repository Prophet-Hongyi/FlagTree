add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  TensorMemoryUtils.cpp
  Ops.cpp

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)

if(BUILD_MCTLE)
  add_dependencies(TritonNvidiaGPUIR McTleTableGen)
endif()

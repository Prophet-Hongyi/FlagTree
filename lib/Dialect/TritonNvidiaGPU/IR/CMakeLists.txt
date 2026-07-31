add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  TMAEncodingUtils.cpp
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

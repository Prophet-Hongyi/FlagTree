# Copyright 2026- Xcoresigma Technology Co., Ltd

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
# TileIR dialect (FLIR) headers — tle_ir.cc creates tile.* ops
include_directories(${CMAKE_SOURCE_DIR}/third_party/FLUTE/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/FLUTE/include)
add_subdirectory(dialect)


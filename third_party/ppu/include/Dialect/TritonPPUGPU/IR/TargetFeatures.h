/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_DIALECT_TRITONPPUGPU_IR_TARGETFEATURES_H
#define TRITON_DIALECT_TRITONPPUGPU_IR_TARGETFEATURES_H

namespace mlir::triton::ppu {

enum class LowPrecisionMode { Unsupported, Software, Native };

constexpr const char *stringifyLowPrecisionMode(LowPrecisionMode mode) {
  switch (mode) {
  case LowPrecisionMode::Unsupported:
    return "unsupported";
  case LowPrecisionMode::Software:
    return "software";
  case LowPrecisionMode::Native:
    return "native";
  }
  return "unsupported";
}

// Single compiler-routing source of truth for PPU low-precision features.
// "Native" only describes the selected compiler path. Device qualification is
// a separate selector and hardware-evidence gate.
class TargetFeatures {
public:
  explicit constexpr TargetFeatures(int computeCapability)
      : computeCapability(computeCapability) {}

  constexpr bool isPPU0010() const { return computeCapability == 80; }
  constexpr bool isPPU0015() const { return computeCapability == 89; }
  constexpr bool isKnownArchitecture() const {
    return isPPU0010() || isPPU0015();
  }

  constexpr const char *getArchitectureName() const {
    if (isPPU0010())
      return "ppu0010";
    if (isPPU0015())
      return "ppu0015";
    return "unknown";
  }

  constexpr LowPrecisionMode getFp8ConversionMode() const {
    if (isPPU0010())
      return LowPrecisionMode::Software;
    if (isPPU0015())
      return LowPrecisionMode::Native;
    return LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getFp8MmaMode() const {
    if (isPPU0010())
      return LowPrecisionMode::Software;
    if (isPPU0015())
      return LowPrecisionMode::Native;
    return LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getSignedInt8MmaMode() const {
    if (isKnownArchitecture())
      return LowPrecisionMode::Native;
    return LowPrecisionMode::Unsupported;
  }

  constexpr bool supportsFp8Dtypes() const {
    return getFp8ConversionMode() != LowPrecisionMode::Unsupported;
  }

  constexpr bool supportsBatchedDotScaled() const {
    return isKnownArchitecture();
  }

private:
  int computeCapability;
};

} // namespace mlir::triton::ppu

#endif // TRITON_DIALECT_TRITONPPUGPU_IR_TARGETFEATURES_H

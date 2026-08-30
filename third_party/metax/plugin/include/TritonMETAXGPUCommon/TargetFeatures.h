#ifndef TRITONMETAXGPU_COMMON_TARGET_FEATURES_H
#define TRITONMETAXGPU_COMMON_TARGET_FEATURES_H

namespace mlir::triton::metax {

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

// Single compiler-routing source of truth for MetaX low-precision features.
// Native describes a qualified compiler path for an exact architecture.  It
// must not be inferred from CUDA-like numeric capability ordering.
class TargetFeatures {
public:
  explicit constexpr TargetFeatures(int computeCapability)
      : computeCapability(computeCapability) {}

  constexpr bool isC550() const { return computeCapability == 80; }

  constexpr const char *getArchitectureName() const {
    return isC550() ? "c550" : "unknown";
  }

  constexpr LowPrecisionMode getOcpFp8ConversionMode() const {
    return isC550() ? LowPrecisionMode::Software
                    : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getCustomFp8ConversionMode() const {
    return isC550() ? LowPrecisionMode::Software
                    : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getFp8MmaMode() const {
    return isC550() ? LowPrecisionMode::Software
                    : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getFp4ConversionMode() const {
    return isC550() ? LowPrecisionMode::Software
                    : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getSignedInt8MmaMode() const {
    return isC550() ? LowPrecisionMode::Native : LowPrecisionMode::Unsupported;
  }

  constexpr bool supportsFp8Conversion() const {
    return getOcpFp8ConversionMode() != LowPrecisionMode::Unsupported;
  }

  constexpr bool supportsSoftwareFp8Dot() const {
    return getFp8MmaMode() == LowPrecisionMode::Software;
  }

  constexpr bool supportsSoftwareFp4DotScaled() const {
    return getFp4ConversionMode() == LowPrecisionMode::Software;
  }

private:
  int computeCapability;
};

} // namespace mlir::triton::metax

#endif // TRITONMETAXGPU_COMMON_TARGET_FEATURES_H

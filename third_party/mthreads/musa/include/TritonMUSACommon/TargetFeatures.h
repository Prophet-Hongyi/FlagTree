#ifndef TRITONMUSA_COMMON_TARGET_FEATURES_H
#define TRITONMUSA_COMMON_TARGET_FEATURES_H

namespace mlir::triton::musa {

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

// Single compiler-routing source of truth for MUSA low-precision features.
// Native describes the selected compiler path for an exact architecture; the
// device, toolchain and generated-binary qualification remains a separate gate.
class TargetFeatures {
public:
  explicit constexpr TargetFeatures(int computeCapability)
      : computeCapability(computeCapability) {}

  constexpr bool isPH1() const { return computeCapability == 31; }

  constexpr const char *getArchitectureName() const {
    return isPH1() ? "ph1" : "unknown";
  }

  constexpr LowPrecisionMode getOcpFp8ConversionMode() const {
    return isPH1() ? LowPrecisionMode::Native : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getCustomFp8ConversionMode() const {
    return isPH1() ? LowPrecisionMode::Software : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getFp8MmaMode() const {
    return isPH1() ? LowPrecisionMode::Native : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getFp4ConversionMode() const {
    return isPH1() ? LowPrecisionMode::Software : LowPrecisionMode::Unsupported;
  }

  constexpr LowPrecisionMode getSignedInt8MmaMode() const {
    return isPH1() ? LowPrecisionMode::Native : LowPrecisionMode::Unsupported;
  }

  constexpr bool supportsMmaLowering() const { return isPH1(); }
  constexpr bool supportsBatchedDotScaled() const { return isPH1(); }

private:
  int computeCapability;
};

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_TARGET_FEATURES_H

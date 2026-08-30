#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/TargetParser.h"

namespace mlir::triton::HCU {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
  RDNA4,
  GFX1250,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);
ISAFamily deduceISAFamily(llvm::AMDGPU::GPUKind kind);

// Retursn true if given architecture support V_DOT instruction.
bool supportsVDot(llvm::StringRef arch);

bool isCDNA(ISAFamily isaFamily);

bool isRDNA(ISAFamily isaFamily);

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/8c75290/llvm/lib/Target/HCUGPU/SIDefines.h#L939
enum class DppCtrl : uint32_t {
  QUAD_PERM_FIRST = 0,
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  BCAST15 = 0x142,
  BCAST31 = 0x143
};

// HCU ISA features
enum class HCUISAFeature : uint64_t {
  NONE = 0,
  MMAC_LAYOUT = 1 << 0,
  MAMC_FP8 = 1 << 1,
  MMAC_FP6FP4 = 1 << 2,
  MMAC_SCALE = 1 << 3,
  MLS = 1 << 4,
  CVT_FP8F32 = 1 << 5,
  CVT_FP8F16 = 1 << 6,
};

inline constexpr bool operator&(HCUISAFeature lhs, HCUISAFeature rhs) {
  return static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs);
}
inline constexpr HCUISAFeature operator|(HCUISAFeature lhs, HCUISAFeature rhs) {
  return static_cast<HCUISAFeature>(static_cast<uint64_t>(lhs) |
                                    static_cast<uint64_t>(rhs));
}
HCUISAFeature deduceHCUISAFeature(llvm::StringRef arch);
HCUISAFeature deduceHCUISAFeature(llvm::AMDGPU::GPUKind kind);
bool supportsHCUISAFeature(llvm::StringRef arch, HCUISAFeature feature);

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

// Compiler-routing source of truth for low-precision paths.  Product claims
// remain exact-target claims: gfx936 identifies the qualified BW1000 path;
// unrelated numeric gfx values must not inherit its capabilities.
class LowPrecisionTargetFeatures {
public:
  explicit LowPrecisionTargetFeatures(llvm::StringRef arch);
  explicit LowPrecisionTargetFeatures(llvm::AMDGPU::GPUKind kind)
      : kind(kind) {}

  bool isBW1000() const;
  bool isKnownTarget() const;
  const char *getArchitectureName() const;

  LowPrecisionMode getOcpFp8ConversionMode() const;
  LowPrecisionMode getCustomFp8ConversionMode() const;
  LowPrecisionMode getFp8MmaMode() const;
  LowPrecisionMode getFp4ConversionMode() const;
  LowPrecisionMode getFp4MmaMode() const;
  LowPrecisionMode getSignedInt8MmaMode() const;

  bool supportsSoftwareFp8Dot() const {
    return getFp8MmaMode() == LowPrecisionMode::Software;
  }
  bool supportsSoftwareFp4DotScaled() const {
    return getFp4MmaMode() == LowPrecisionMode::Software;
  }

private:
  llvm::AMDGPU::GPUKind kind;
};

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_TARGETUTILS_H_

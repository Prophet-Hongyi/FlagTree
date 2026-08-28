// Sunrise owns a backend-spec replacement for TritonGPUTransforms, but the RLC
// algorithm is shared with core. Keep this translation-unit shim so the vendor
// source overlay still owns the target while compiling one common algorithm.
// Sunrise-specific truncation legality is injected through RlcBackendPolicy.
#define __FLAGTREE_SUNRISE_RLC__
#include "../../../../../../../lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp"

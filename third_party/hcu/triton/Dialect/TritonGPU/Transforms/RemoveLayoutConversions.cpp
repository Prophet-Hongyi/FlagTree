// HCU owns a backend-local TritonGPUTransforms target and generated pass
// headers, but the RLC algorithm is shared with the core implementation. Keep
// this translation-unit shim so the vendor build retains that ownership while
// compiling the portable phases against the HCU dialect definitions.
// FLAGTREE_TLE stays on for HCU, so the public file keeps tle.remote_pointers
// guards. Do not fold add_hoist_layout_conversions into this file.
#define __FLAGTREE_HCU_RLC__
#include "../../../../../../lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp"

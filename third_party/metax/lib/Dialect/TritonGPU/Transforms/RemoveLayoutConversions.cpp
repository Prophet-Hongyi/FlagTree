// MetaX owns a backend-local TritonGPUTransforms target and generated pass
// headers, but the RLC algorithm is shared with the core implementation. Keep
// this translation-unit shim so the vendor build retains that ownership while
// compiling the portable phases against the MetaX dialect definitions.
// NVIDIA TLE remat stays inactive: FLAGTREE_BACKEND=metax turns FLAGTREE_TLE
// off and uses MCTLE instead. Do not fold MetaX change_layout_* passes into
// this file.
#define __FLAGTREE_METAX_RLC__
#include "../../../../../../lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp"

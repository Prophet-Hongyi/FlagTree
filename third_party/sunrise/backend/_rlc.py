"""Sunrise phased-RLC validation and cache identity helpers."""

from collections.abc import Mapping


SUPPORTED_RLC_ARCHES = frozenset({"s2", "s3"})

RLC_POLICY_FIELDS = (
    ("rlc_minimum_writeback_bits", "ttg.rlc-minimum-writeback-bits"),
    ("rlc_convert_minimum_elements", "ttg.rlc-convert-minimum-elements"),
    ("rlc_convert_minimum_element_bits", "ttg.rlc-convert-minimum-element-bits"),
    ("rlc_convert_cost_per_byte", "ttg.rlc-convert-cost-per-byte"),
    ("rlc_cached_load_cost_per_byte", "ttg.rlc-cached-load-cost-per-byte"),
    ("rlc_expensive_math_cost_per_byte", "ttg.rlc-expensive-math-cost-per-byte"),
    ("rlc_inter_warp_reduce_cost", "ttg.rlc-inter-warp-reduce-cost"),
)

RLC_OPTION_NAMES = ("rlc_enhance", "rlc_phase_mask", *(name for name, _ in RLC_POLICY_FIELDS))


def _read(config, name):
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)


def normalize_rlc_arch(arch):
    return str(arch).lower()


def validate_rlc_phase_mask(value):
    mask = int(value)
    if mask < 0 or mask & ~0xF:
        raise ValueError(f"Sunrise RLC phase mask must be in [0, 15], got {value!r}")
    return mask


def validate_rlc_arch(arch, enhance):
    normalized = normalize_rlc_arch(arch)
    if enhance and normalized not in SUPPORTED_RLC_ARCHES:
        supported = ", ".join(sorted(SUPPORTED_RLC_ARCHES))
        raise ValueError(f"Sunrise enhanced RLC supports only {supported}; got {arch!r}")
    return normalized


def rlc_policy_signature(config):
    values = [
        str(int(bool(_read(config, "rlc_enhance")))),
        str(validate_rlc_phase_mask(_read(config, "rlc_phase_mask"))),
    ]
    for name, _ in RLC_POLICY_FIELDS:
        value = int(_read(config, name))
        if value < 0:
            raise ValueError(f"Sunrise RLC policy override {name} must be non-negative, got {value}")
        values.append(str(value))
    return "-".join(values)

"""Fail-closed LLVM IR bridge for the legacy gfx936 DTK17 toolchain.

FlagTree's LLVM22 ROCDL lowering uses opaque address-space-8 buffer
descriptors and five-operand HCU MMAC intrinsics.  The BW1000 product DTK17
backend instead selects three-operand ``llvm.amdgcn.mmac`` and classic
``<4 x i32>`` raw-buffer intrinsics.  This module bridges only those contracts
at the LLIR-to-assembly boundary; it does not change TTGIR or RLC semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


class LLVM17ContractBridgeError(ValueError):
    """Raised when an input cannot be represented by the DTK17 contract."""


@dataclass(frozen=True)
class LLVM17ContractBridgeStats:
    make_buffer_calls: int = 0
    raw_buffer_load_calls: int = 0
    raw_buffer_store_calls: int = 0
    raw_buffer_atomic_calls: int = 0
    i64_vector_load_materializations: int = 0
    i64_vector_store_materializations: int = 0
    i64_scalar_load_materializations: int = 0
    i64_scalar_store_materializations: int = 0
    resource_phi_materializations: int = 0
    duplicate_declarations_suppressed: int = 0
    mmac_calls: int = 0
    mmac_f16_calls: int = 0
    mmac_bf16_calls: int = 0


_MAKE_NAME = "llvm.amdgcn.make.buffer.rsrc.p8.p1"
_PTR_LOAD = "llvm.amdgcn.raw.ptr.buffer.load"
_RAW_LOAD = "llvm.amdgcn.raw.buffer.load"
_PTR_STORE = "llvm.amdgcn.raw.ptr.buffer.store"
_RAW_STORE = "llvm.amdgcn.raw.buffer.store"
_PTR_ATOMIC = "llvm.amdgcn.raw.ptr.buffer.atomic"
_RAW_ATOMIC = "llvm.amdgcn.raw.buffer.atomic"
_MMAC_VARIANTS = {
    "f16": (
        "llvm.hcu.mmac.f32.16x16x16.f16",
        "llvm.amdgcn.mmac.f32.16x16x16f16",
    ),
    "bf16": (
        "llvm.hcu.mmac.f32.16x16x16.bf16",
        "llvm.amdgcn.mmac.f32.16x16x16bf16",
    ),
}
_I64_VECTOR_PTR_LOAD = _PTR_LOAD + ".v2i64"
_I32_VECTOR_RAW_LOAD = _RAW_LOAD + ".v4i32"
_I64_VECTOR_PTR_STORE = _PTR_STORE + ".v2i64"
_I32_VECTOR_RAW_STORE = _RAW_STORE + ".v4i32"
_I64_SCALAR_PTR_LOAD = _PTR_LOAD + ".i64"
_I32_PAIR_RAW_LOAD = _RAW_LOAD + ".v2i32"
_I64_SCALAR_PTR_STORE = _PTR_STORE + ".i64"
_I32_PAIR_RAW_STORE = _RAW_STORE + ".v2i32"

_MAKE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = (?:tail )?call "
    r"ptr addrspace\(8\) @llvm\.amdgcn\.make\.buffer\.rsrc\.p8\.p1\("
    r"ptr addrspace\(1\) (?:nonnull )?(?P<base>%[-A-Za-z$._0-9]+), "
    r"i16 (?P<stride>-?[0-9]+), i32 (?P<num>-?[0-9]+), "
    r"i32 (?P<flags>-?[0-9]+)\)(?P<suffix>.*)$"
)

_I64_VECTOR_STORE_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<call>(?:tail )?call void )@"
    + re.escape(_I64_VECTOR_PTR_STORE)
    + r"\(<2 x i64> (?P<value>%[-A-Za-z$._0-9]+), "
    r"ptr addrspace\(8\) (?P<resource>%[-A-Za-z$._0-9]+), "
    r"(?P<rest>.*)$"
)

_I64_VECTOR_LOAD_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = "
    r"(?P<call>(?:tail )?call <2 x i64> )@"
    + re.escape(_I64_VECTOR_PTR_LOAD)
    + r"\(ptr addrspace\(8\) (?P<resource>%[-A-Za-z$._0-9]+), "
    r"(?P<rest>.*)$"
)

_I64_SCALAR_STORE_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<call>(?:tail )?call void )@"
    + re.escape(_I64_SCALAR_PTR_STORE)
    + r"\(i64 (?P<value>%[-A-Za-z$._0-9]+), "
    r"ptr addrspace\(8\) (?P<resource>%[-A-Za-z$._0-9]+), "
    r"(?P<rest>.*)$"
)

_I64_SCALAR_LOAD_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = "
    r"(?P<call>(?:tail )?call i64 )@"
    + re.escape(_I64_SCALAR_PTR_LOAD)
    + r"\(ptr addrspace\(8\) (?P<resource>%[-A-Za-z$._0-9]+), "
    r"(?P<rest>.*)$"
)

_RESOURCE_PHI_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = "
    r"phi ptr addrspace\(8\) (?P<incoming>.*)$"
)
_PHI_INCOMING_RE = re.compile(
    r"\[\s*(?P<value>%[-A-Za-z$._0-9]+)\s*,\s*%[-A-Za-z$._0-9]+\s*\]"
)


def _tag_for(result: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", result.lstrip("%"))


def _intrinsic_symbol(line: str, prefix: str) -> str:
    match = re.search(r"@(?P<symbol>" + re.escape(prefix) + r"[^\s(]*)\(", line)
    if match is None:
        raise LLVM17ContractBridgeError(
            f"cannot parse intrinsic symbol for {prefix}: {line}"
        )
    return match.group("symbol")


def _rewrite_resource_operand(line: str) -> str:
    """Materialize the descriptor operand while dropping a known pointer fact."""

    line = line.replace("ptr addrspace(8) nonnull ", "<4 x i32> ")
    return line.replace("ptr addrspace(8) ", "<4 x i32> ")


def _rewrite_resource_descriptor(match: re.Match[str]) -> list[str]:
    stride = int(match.group("stride"))
    if stride != 0:
        raise LLVM17ContractBridgeError(
            "DTK17 bridge cannot encode a nonzero pointer-buffer stride: "
            + match.group(0)
        )

    indent = match.group("indent")
    result = match.group("result")
    base = match.group("base")
    num = match.group("num")
    flags = match.group("flags")
    suffix = match.group("suffix")
    tag = _tag_for(result)
    p64 = f"%flagtree.rsrc.{tag}.base64"
    p128 = f"%flagtree.rsrc.{tag}.base128"
    n128 = f"%flagtree.rsrc.{tag}.num128"
    nsh = f"%flagtree.rsrc.{tag}.numshift"
    f128 = f"%flagtree.rsrc.{tag}.flags128"
    fsh = f"%flagtree.rsrc.{tag}.flagshift"
    low96 = f"%flagtree.rsrc.{tag}.low96"
    packed = f"%flagtree.rsrc.{tag}.packed"
    return [
        f"{indent}{p64} = ptrtoint ptr addrspace(1) {base} to i64",
        f"{indent}{p128} = zext i64 {p64} to i128",
        f"{indent}{n128} = zext i32 {num} to i128",
        f"{indent}{nsh} = shl i128 {n128}, 64",
        f"{indent}{f128} = zext i32 {flags} to i128",
        f"{indent}{fsh} = shl i128 {f128}, 96",
        f"{indent}{low96} = or i128 {p128}, {nsh}",
        f"{indent}{packed} = or i128 {low96}, {fsh}",
        f"{indent}{result} = bitcast i128 {packed} to <4 x i32>{suffix}",
    ]


def bridge_gfx936_llvm17_contract(
    source: str,
    *,
    materialize_i64_vector_loads: bool = False,
    materialize_i64_vector_stores: bool = False,
    materialize_i64_scalar_loads: bool = False,
    materialize_i64_scalar_stores: bool = False,
    materialize_resource_phis: bool = False,
) -> tuple[str, LLVM17ContractBridgeStats]:
    """Translate the supported LLVM22 gfx936 contracts to DTK17 contracts.

    The legacy three-operand MMAC has no LIT/LTS fields.  Only the default
    legacy layout (both flags false) is representable; an interleave/transpose
    request fails instead of silently discarding layout semantics.
    """

    counts = {
        "make": 0,
        "load_call": 0,
        "store_call": 0,
        "atomic_call": 0,
        "i64_vector_load": 0,
        "i64_store": 0,
        "i64_scalar_load": 0,
        "i64_scalar_store": 0,
        "resource_phi": 0,
        "duplicate_decl": 0,
        "mmac_f16_call": 0,
        "mmac_bf16_call": 0,
        "make_decl": 0,
        "load_decl": 0,
        "store_decl": 0,
        "atomic_decl": 0,
        "mmac_f16_decl": 0,
        "mmac_bf16_decl": 0,
    }
    called_overloads = {"load": set(), "store": set(), "atomic": set()}
    declared_overloads = {"load": set(), "store": set(), "atomic": set()}
    output: list[str] = []
    emitted_declarations: dict[str, str] = {}

    def append_line(line: str) -> None:
        """Emit one line while accepting only exact duplicate declarations."""

        if line.lstrip().startswith("declare "):
            match = re.search(r"@(?P<symbol>[^\s(]+)\(", line)
            if match is None:
                raise LLVM17ContractBridgeError(
                    "cannot parse bridged declaration: " + line
                )
            symbol = match.group("symbol")
            normalized = line.strip()
            previous = emitted_declarations.get(symbol)
            if previous is not None:
                if previous != normalized:
                    raise LLVM17ContractBridgeError(
                        "conflicting bridged declarations for "
                        f"{symbol}: first={previous!r} second={normalized!r}"
                    )
                counts["duplicate_decl"] += 1
                return
            emitted_declarations[symbol] = normalized
        output.append(line)
    materialized_resources = {
        match.group("result")
        for line in source.splitlines()
        if (match := _MAKE_RE.match(line)) is not None
    }

    for original_line in source.splitlines():
        line = original_line
        make_match = _MAKE_RE.match(line)
        if make_match:
            counts["make"] += 1
            output.extend(_rewrite_resource_descriptor(make_match))
            continue

        if line.startswith("declare ptr addrspace(8) @" + _MAKE_NAME):
            counts["make_decl"] += 1
            continue

        resource_phi = _RESOURCE_PHI_RE.match(line)
        if resource_phi and materialize_resource_phis:
            incoming_values = [
                match.group("value")
                for match in _PHI_INCOMING_RE.finditer(resource_phi.group("incoming"))
            ]
            if not incoming_values or any(
                value not in materialized_resources for value in incoming_values
            ):
                raise LLVM17ContractBridgeError(
                    "gfx936 resource phi has a non-descriptor incoming value: "
                    + original_line
                )
            counts["resource_phi"] += 1
            materialized_resources.add(resource_phi.group("result"))
            output.append(line.replace("phi ptr addrspace(8)", "phi <4 x i32>", 1))
            continue

        if materialize_i64_vector_loads and "@" + _I64_VECTOR_PTR_LOAD in line:
            if line.lstrip().startswith("declare "):
                counts["load_decl"] += 1
                declared_overloads["load"].add(_I64_VECTOR_PTR_LOAD)
                line = line.replace(
                    "declare <2 x i64> @" + _I64_VECTOR_PTR_LOAD,
                    "declare <4 x i32> @" + _I32_VECTOR_RAW_LOAD,
                    1,
                )
                line = line.replace(
                    "ptr addrspace(8) readonly nocapture", "<4 x i32>", 1
                )
                append_line(line)
                continue

            vector_load = _I64_VECTOR_LOAD_CALL_RE.match(line)
            if vector_load is None:
                raise LLVM17ContractBridgeError(
                    "cannot materialize gfx936 v2i64 raw-buffer load: "
                    + original_line
                )
            counts["load_call"] += 1
            counts["i64_vector_load"] += 1
            called_overloads["load"].add(_I64_VECTOR_PTR_LOAD)
            result = vector_load.group("result")
            resource = vector_load.group("resource")
            temporary = (
                f"%flagtree.i64loadvec.{counts['i64_vector_load']}.v4i32"
            )
            output.append(
                f"{vector_load.group('indent')}{temporary} = "
                f"{vector_load.group('call').replace('call <2 x i64> ', 'call <4 x i32> ', 1)}"
                f"@{_I32_VECTOR_RAW_LOAD}(<4 x i32> {resource}, "
                f"{vector_load.group('rest')}"
            )
            output.append(
                f"{vector_load.group('indent')}{result} = "
                f"bitcast <4 x i32> {temporary} to <2 x i64>"
            )
            continue

        if materialize_i64_scalar_loads and "@" + _I64_SCALAR_PTR_LOAD in line:
            if line.lstrip().startswith("declare "):
                counts["load_decl"] += 1
                declared_overloads["load"].add(_I64_SCALAR_PTR_LOAD)
                line = line.replace(
                    "declare i64 @" + _I64_SCALAR_PTR_LOAD,
                    "declare <2 x i32> @" + _I32_PAIR_RAW_LOAD,
                    1,
                )
                line = line.replace(
                    "ptr addrspace(8) readonly nocapture", "<4 x i32>", 1
                )
                append_line(line)
                continue

            scalar_load = _I64_SCALAR_LOAD_CALL_RE.match(line)
            if scalar_load is None:
                raise LLVM17ContractBridgeError(
                    "cannot materialize gfx936 i64 raw-buffer load: "
                    + original_line
                )
            counts["load_call"] += 1
            counts["i64_scalar_load"] += 1
            called_overloads["load"].add(_I64_SCALAR_PTR_LOAD)
            result = scalar_load.group("result")
            resource = scalar_load.group("resource")
            temporary = (
                f"%flagtree.i64load.{counts['i64_scalar_load']}.v2i32"
            )
            output.append(
                f"{scalar_load.group('indent')}{temporary} = "
                f"{scalar_load.group('call').replace('call i64 ', 'call <2 x i32> ', 1)}"
                f"@{_I32_PAIR_RAW_LOAD}(<4 x i32> {resource}, "
                f"{scalar_load.group('rest')}"
            )
            output.append(
                f"{scalar_load.group('indent')}{result} = "
                f"bitcast <2 x i32> {temporary} to i64"
            )
            continue

        if materialize_i64_scalar_stores and "@" + _I64_SCALAR_PTR_STORE in line:
            if line.lstrip().startswith("declare "):
                counts["store_decl"] += 1
                declared_overloads["store"].add(_I64_SCALAR_PTR_STORE)
                line = line.replace(
                    "@" + _I64_SCALAR_PTR_STORE, "@" + _I32_PAIR_RAW_STORE
                )
                line = line.replace("(i64,", "(<2 x i32>,", 1)
                line = line.replace(
                    "ptr addrspace(8) writeonly nocapture", "<4 x i32>"
                )
                append_line(line)
                continue

            scalar_store = _I64_SCALAR_STORE_CALL_RE.match(line)
            if scalar_store is None:
                raise LLVM17ContractBridgeError(
                    "cannot materialize gfx936 i64 raw-buffer store: "
                    + original_line
                )
            counts["store_call"] += 1
            counts["i64_scalar_store"] += 1
            called_overloads["store"].add(_I64_SCALAR_PTR_STORE)
            value = scalar_store.group("value")
            resource = scalar_store.group("resource")
            cast = f"%flagtree.i64scalar.{counts['i64_scalar_store']}.v2i32"
            output.append(
                f"{scalar_store.group('indent')}{cast} = "
                f"bitcast i64 {value} to <2 x i32>"
            )
            output.append(
                f"{scalar_store.group('indent')}{scalar_store.group('call')}"
                f"@{_I32_PAIR_RAW_STORE}(<2 x i32> {cast}, "
                f"<4 x i32> {resource}, {scalar_store.group('rest')}"
            )
            continue

        if materialize_i64_vector_stores and "@" + _I64_VECTOR_PTR_STORE in line:
            if line.lstrip().startswith("declare "):
                counts["store_decl"] += 1
                declared_overloads["store"].add(_I64_VECTOR_PTR_STORE)
                line = line.replace(
                    "@" + _I64_VECTOR_PTR_STORE, "@" + _I32_VECTOR_RAW_STORE
                )
                line = line.replace("<2 x i64>", "<4 x i32>", 1)
                line = line.replace(
                    "ptr addrspace(8) writeonly nocapture", "<4 x i32>"
                )
                append_line(line)
                continue

            store_match = _I64_VECTOR_STORE_CALL_RE.match(line)
            if store_match is None:
                raise LLVM17ContractBridgeError(
                    "cannot materialize gfx936 v2i64 raw-buffer store: "
                    + original_line
                )
            counts["store_call"] += 1
            counts["i64_store"] += 1
            called_overloads["store"].add(_I64_VECTOR_PTR_STORE)
            value = store_match.group("value")
            resource = store_match.group("resource")
            cast = f"%flagtree.i64store.{counts['i64_store']}.v4i32"
            output.append(
                f"{store_match.group('indent')}{cast} = "
                f"bitcast <2 x i64> {value} to <4 x i32>"
            )
            output.append(
                f"{store_match.group('indent')}{store_match.group('call')}"
                f"@{_I32_VECTOR_RAW_STORE}(<4 x i32> {cast}, "
                f"<4 x i32> {resource}, {store_match.group('rest')}"
            )
            continue

        if "@" + _PTR_LOAD in line:
            load_symbol = _intrinsic_symbol(line, _PTR_LOAD)
            line = line.replace("@" + _PTR_LOAD, "@" + _RAW_LOAD)
            if line.lstrip().startswith("declare "):
                counts["load_decl"] += 1
                declared_overloads["load"].add(load_symbol)
                line = line.replace(
                    "ptr addrspace(8) readonly nocapture", "<4 x i32>"
                )
            else:
                counts["load_call"] += 1
                called_overloads["load"].add(load_symbol)
                line = _rewrite_resource_operand(line)

        if "@" + _PTR_STORE in line:
            store_symbol = _intrinsic_symbol(line, _PTR_STORE)
            line = line.replace("@" + _PTR_STORE, "@" + _RAW_STORE)
            if line.lstrip().startswith("declare "):
                counts["store_decl"] += 1
                declared_overloads["store"].add(store_symbol)
                line = line.replace(
                    "ptr addrspace(8) writeonly nocapture", "<4 x i32>"
                )
            else:
                counts["store_call"] += 1
                called_overloads["store"].add(store_symbol)
                line = _rewrite_resource_operand(line)

        if "@" + _PTR_ATOMIC in line:
            atomic_symbol = _intrinsic_symbol(line, _PTR_ATOMIC)
            line = line.replace("@" + _PTR_ATOMIC, "@" + _RAW_ATOMIC)
            if line.lstrip().startswith("declare "):
                counts["atomic_decl"] += 1
                declared_overloads["atomic"].add(atomic_symbol)
                line = line.replace(
                    "ptr addrspace(8) nocapture", "<4 x i32>"
                )
            else:
                counts["atomic_call"] += 1
                called_overloads["atomic"].add(atomic_symbol)
                line = _rewrite_resource_operand(line)

        mmac_variant = next(
            (
                (kind, hcu_mmac, dtk_mmac)
                for kind, (hcu_mmac, dtk_mmac) in _MMAC_VARIANTS.items()
                if "@" + hcu_mmac in line
            ),
            None,
        )
        if mmac_variant is not None:
            kind, hcu_mmac, dtk_mmac = mmac_variant
            line = line.replace("@" + hcu_mmac, "@" + dtk_mmac)
            if line.lstrip().startswith("declare "):
                counts[f"mmac_{kind}_decl"] += 1
                line, replacements = re.subn(r", i1, i1\)(?=\s*#)", ")", line)
            else:
                flag_match = re.search(
                    r", i1 (?P<lit>true|false), i1 (?P<lts>true|false)"
                    r"\)(?=,?\s*!dbg)",
                    line,
                )
                if flag_match is None:
                    raise LLVM17ContractBridgeError(
                        "cannot parse HCU MMAC LIT/LTS flags: " + original_line
                    )
                if flag_match.group("lit") != "false" or flag_match.group("lts") != "false":
                    raise LLVM17ContractBridgeError(
                        "DTK17 three-operand MMAC cannot represent enabled LIT/LTS: "
                        + original_line
                    )
                counts[f"mmac_{kind}_call"] += 1
                line, replacements = re.subn(
                    r", i1 false, i1 false\)(?=,?\s*!dbg)", ")", line
                )
            if replacements != 1:
                raise LLVM17ContractBridgeError(
                    "HCU MMAC signature rewrite failed: " + original_line
                )

        append_line(line)

    if counts["make"] > 0 and counts["make_decl"] != 1:
        raise LLVM17ContractBridgeError(f"make-buffer declaration mismatch: {counts}")
    for kind in ("load", "store", "atomic"):
        missing = called_overloads[kind] - declared_overloads[kind]
        if missing:
            raise LLVM17ContractBridgeError(
                f"raw-buffer {kind} overloads have no matching declaration: "
                f"missing={sorted(missing)} counts={counts}"
            )
    for kind in _MMAC_VARIANTS:
        if counts[f"mmac_{kind}_call"] > 0 and counts[f"mmac_{kind}_decl"] != 1:
            raise LLVM17ContractBridgeError(
                f"{kind} MMAC declaration mismatch: {counts}"
            )

    result = "\n".join(output) + ("\n" if source.endswith("\n") else "")
    forbidden = (
        _MAKE_NAME,
        _PTR_LOAD,
        _PTR_STORE,
        _PTR_ATOMIC,
        "llvm.hcu.mmac",
        "ptr addrspace(8)",
    )
    for contract in forbidden:
        if contract in result:
            raise LLVM17ContractBridgeError(
                f"unsupported or unbridged LLVM17 contract remains: {contract}"
            )

    stats = LLVM17ContractBridgeStats(
        make_buffer_calls=counts["make"],
        raw_buffer_load_calls=counts["load_call"],
        raw_buffer_store_calls=counts["store_call"],
        raw_buffer_atomic_calls=counts["atomic_call"],
        i64_vector_load_materializations=counts["i64_vector_load"],
        i64_vector_store_materializations=counts["i64_store"],
        i64_scalar_load_materializations=counts["i64_scalar_load"],
        i64_scalar_store_materializations=counts["i64_scalar_store"],
        resource_phi_materializations=counts["resource_phi"],
        duplicate_declarations_suppressed=counts["duplicate_decl"],
        mmac_calls=counts["mmac_f16_call"] + counts["mmac_bf16_call"],
        mmac_f16_calls=counts["mmac_f16_call"],
        mmac_bf16_calls=counts["mmac_bf16_call"],
    )
    return result, stats

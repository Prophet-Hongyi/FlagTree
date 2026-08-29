"""Focused LLVM17 compatibility bridges for gfx936 kernels.

The in-tree LLVM lowering emits the newer HCU MMAC contract and opaque
address-space-8 buffer descriptors.  The product DTK17 toolchain selects the
same native instructions through legacy MMAC intrinsics and classic
``<4 x i32>`` raw-buffer descriptors.  This module bridges only explicitly
listed contracts and rejects every other overload.
"""

from dataclasses import dataclass
import re


class LLVM17MmacBridgeError(ValueError):
    """Raised when a new MMAC contract cannot be represented by DTK17."""


@dataclass(frozen=True)
class LLVM17MmacBridgeStats:
    calls: int = 0
    make_buffer_calls: int = 0
    raw_buffer_load_calls: int = 0
    raw_buffer_store_calls: int = 0


NEW_INT8_MMAC = "llvm.hcu.mmac.i32.16x16x32.i8"
LEGACY_INT8_MMAC = "llvm.amdgcn.mmac.i32.16x16x32i8"
NEW_FP16_MMAC = "llvm.hcu.mmac.f32.16x16x16.f16"
LEGACY_FP16_MMAC = "llvm.amdgcn.mmac.f32.16x16x16f16"
MAKE_BUFFER_RSRC = "llvm.amdgcn.make.buffer.rsrc.p8.p1"
PTR_BUFFER_LOAD_I16 = "llvm.amdgcn.raw.ptr.buffer.load.i16"
RAW_BUFFER_LOAD_I16 = "llvm.amdgcn.raw.buffer.load.i16"
PTR_BUFFER_LOAD_I32 = "llvm.amdgcn.raw.ptr.buffer.load.i32"
RAW_BUFFER_LOAD_I32 = "llvm.amdgcn.raw.buffer.load.i32"
PTR_BUFFER_LOAD_I8 = "llvm.amdgcn.raw.ptr.buffer.load.i8"
RAW_BUFFER_LOAD_I8 = "llvm.amdgcn.raw.buffer.load.i8"
PTR_BUFFER_LOAD_F32 = "llvm.amdgcn.raw.ptr.buffer.load.f32"
RAW_BUFFER_LOAD_F32 = "llvm.amdgcn.raw.buffer.load.f32"
PTR_BUFFER_LOAD_V4F32 = "llvm.amdgcn.raw.ptr.buffer.load.v4f32"
RAW_BUFFER_LOAD_V4F32 = "llvm.amdgcn.raw.buffer.load.v4f32"
PTR_BUFFER_LOAD_V4I32 = "llvm.amdgcn.raw.ptr.buffer.load.v4i32"
RAW_BUFFER_LOAD_V4I32 = "llvm.amdgcn.raw.buffer.load.v4i32"
PTR_BUFFER_STORE_I32 = "llvm.amdgcn.raw.ptr.buffer.store.i32"
RAW_BUFFER_STORE_I32 = "llvm.amdgcn.raw.buffer.store.i32"
PTR_BUFFER_STORE_V4I32 = "llvm.amdgcn.raw.ptr.buffer.store.v4i32"
RAW_BUFFER_STORE_V4I32 = "llvm.amdgcn.raw.buffer.store.v4i32"
PTR_BUFFER_STORE_I8 = "llvm.amdgcn.raw.ptr.buffer.store.i8"
RAW_BUFFER_STORE_I8 = "llvm.amdgcn.raw.buffer.store.i8"
PTR_BUFFER_STORE_F32 = "llvm.amdgcn.raw.ptr.buffer.store.f32"
RAW_BUFFER_STORE_F32 = "llvm.amdgcn.raw.buffer.store.f32"
PTR_BUFFER_STORE_V4F32 = "llvm.amdgcn.raw.ptr.buffer.store.v4f32"
RAW_BUFFER_STORE_V4F32 = "llvm.amdgcn.raw.buffer.store.v4f32"

# Compatibility names retained for the signed INT8 unit tests and callers.
LLVM17Int8MmacBridgeError = LLVM17MmacBridgeError
LLVM17Int8MmacBridgeStats = LLVM17MmacBridgeStats

_PTR_BUFFER_LOAD_PREFIX = "llvm.amdgcn.raw.ptr.buffer.load"
_PTR_BUFFER_STORE_PREFIX = "llvm.amdgcn.raw.ptr.buffer.store"

_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = "
    r"(?P<tail>tail )?call <4 x i32> @" + re.escape(NEW_INT8_MMAC) + r"\("
    r"<2 x i32> (?P<lhs>%[-A-Za-z$._0-9]+), "
    r"<2 x i32> (?P<rhs>%[-A-Za-z$._0-9]+), "
    r"<4 x i32> (?P<acc>[^,]+), "
    r"i1 (?P<lit>[^,]+), i1 (?P<clamp>[^,]+), i1 (?P<lts>[^)]+)\)"
    r"(?P<suffix>.*)$"
)

_DECL_RE = re.compile(
    r"^(?P<indent>\s*)declare <4 x i32> @" + re.escape(NEW_INT8_MMAC)
    + r"\(<2 x i32>, <2 x i32>, <4 x i32>, "
    r"i1(?: immarg)?, i1(?: immarg)?, i1(?: immarg)?\)(?P<suffix>.*)$"
)

_MAKE_BUFFER_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = (?:tail )?call "
    r"ptr addrspace\(8\) @" + re.escape(MAKE_BUFFER_RSRC) + r"\("
    r"ptr addrspace\(1\) (?:nonnull )?(?P<base>%[-A-Za-z$._0-9]+), "
    r"i16 (?P<stride>-?[0-9]+), i32 (?P<num>-?[0-9]+), "
    r"i32 (?P<flags>-?[0-9]+)\)(?P<suffix>.*)$"
)

_RESOURCE_TYPE_RE = re.compile(
    r"ptr addrspace\(8\)(?: (?:readonly|writeonly|nocapture|nonnull))*"
)


def _tag_for(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value.lstrip("%"))


def _rewrite_resource_descriptor(match: re.Match[str]) -> list[str]:
    if int(match.group("stride")) != 0:
        raise LLVM17MmacBridgeError(
            "DTK17 MMAC bridge cannot encode a nonzero buffer stride: "
            + match.group(0)
        )

    indent = match.group("indent")
    result = match.group("result")
    base = match.group("base")
    num = match.group("num")
    flags = match.group("flags")
    suffix = match.group("suffix")
    tag = _tag_for(result)
    base64 = f"%flagtree.rsrc.{tag}.base64"
    base128 = f"%flagtree.rsrc.{tag}.base128"
    num128 = f"%flagtree.rsrc.{tag}.num128"
    num_shifted = f"%flagtree.rsrc.{tag}.numshift"
    flags128 = f"%flagtree.rsrc.{tag}.flags128"
    flags_shifted = f"%flagtree.rsrc.{tag}.flagshift"
    low96 = f"%flagtree.rsrc.{tag}.low96"
    packed = f"%flagtree.rsrc.{tag}.packed"
    return [
        f"{indent}{base64} = ptrtoint ptr addrspace(1) {base} to i64",
        f"{indent}{base128} = zext i64 {base64} to i128",
        f"{indent}{num128} = zext i32 {num} to i128",
        f"{indent}{num_shifted} = shl i128 {num128}, 64",
        f"{indent}{flags128} = zext i32 {flags} to i128",
        f"{indent}{flags_shifted} = shl i128 {flags128}, 96",
        f"{indent}{low96} = or i128 {base128}, {num_shifted}",
        f"{indent}{packed} = or i128 {low96}, {flags_shifted}",
        f"{indent}{result} = bitcast i128 {packed} to <4 x i32>{suffix}",
    ]


def _replace_resource_type(line: str, contract: str) -> str:
    rewritten, replacements = _RESOURCE_TYPE_RE.subn("<4 x i32>", line, count=1)
    if replacements != 1:
        raise LLVM17MmacBridgeError(
            f"cannot rewrite {contract} resource operand: {line}"
        )
    return rewritten


def _bridge_buffer_contracts(
    source: str,
    *,
    load_contracts: dict[str, str],
    store_contracts: dict[str, str],
    label: str,
) -> tuple[str, LLVM17MmacBridgeStats]:
    allowed = {
        _PTR_BUFFER_LOAD_PREFIX: set(load_contracts),
        _PTR_BUFFER_STORE_PREFIX: set(store_contracts),
    }
    for prefix, allowed_symbols in allowed.items():
        symbols = set(re.findall(r"@(" + re.escape(prefix) + r"[^\s(]*)\(", source))
        unexpected = symbols - allowed_symbols
        if unexpected:
            raise LLVM17MmacBridgeError(
                f"unsupported gfx936 DTK17 {label} buffer overloads: "
                + ", ".join(sorted(unexpected))
            )

    output: list[str] = []
    make_buffer_calls = 0
    make_buffer_declarations = 0
    load_calls = {symbol: 0 for symbol in load_contracts}
    load_declarations = {symbol: 0 for symbol in load_contracts}
    store_calls = {symbol: 0 for symbol in store_contracts}
    store_declarations = {symbol: 0 for symbol in store_contracts}

    for original_line in source.splitlines():
        make_buffer = _MAKE_BUFFER_RE.match(original_line)
        if make_buffer is not None:
            make_buffer_calls += 1
            output.extend(_rewrite_resource_descriptor(make_buffer))
            continue

        if original_line.lstrip().startswith(
            "declare ptr addrspace(8) @" + MAKE_BUFFER_RSRC
        ):
            make_buffer_declarations += 1
            continue

        rewritten_buffer = False
        for contracts, calls, declarations in (
            (load_contracts, load_calls, load_declarations),
            (store_contracts, store_calls, store_declarations),
        ):
            for pointer_symbol, raw_symbol in contracts.items():
                if "@" + pointer_symbol + "(" not in original_line:
                    continue
                rewritten = original_line.replace(
                    "@" + pointer_symbol, "@" + raw_symbol
                )
                if original_line.lstrip().startswith("declare "):
                    declarations[pointer_symbol] += 1
                else:
                    calls[pointer_symbol] += 1
                output.append(_replace_resource_type(rewritten, pointer_symbol))
                rewritten_buffer = True
                break
            if rewritten_buffer:
                break
        if rewritten_buffer:
            continue

        output.append(original_line)

    if make_buffer_calls and make_buffer_declarations != 1:
        raise LLVM17MmacBridgeError(
            f"expected one {label} make-buffer declaration when calls are present; "
            f"got {make_buffer_declarations}"
        )
    for calls_by_symbol, declarations_by_symbol in (
        (load_calls, load_declarations),
        (store_calls, store_declarations),
    ):
        for symbol, calls in calls_by_symbol.items():
            declarations = declarations_by_symbol[symbol]
            if calls and declarations != 1:
                raise LLVM17MmacBridgeError(
                    f"expected one {symbol} declaration when calls are present; "
                    f"got {declarations}"
                )

    bridged = "\n".join(output)
    if source.endswith("\n"):
        bridged += "\n"
    for contract in (MAKE_BUFFER_RSRC, "llvm.amdgcn.raw.ptr.buffer", "ptr addrspace(8)"):
        if contract in bridged:
            raise LLVM17MmacBridgeError(
                f"unresolved gfx936 LLVM17 {label} buffer contract: {contract}"
            )
    return bridged, LLVM17MmacBridgeStats(
        make_buffer_calls=make_buffer_calls,
        raw_buffer_load_calls=sum(load_calls.values()),
        raw_buffer_store_calls=sum(store_calls.values()),
    )


def bridge_gfx936_buffer_contracts_for_llvm17(
    source: str,
) -> tuple[str, LLVM17MmacBridgeStats]:
    """Map observed scalar load/store contracts to DTK17 raw buffers.

    Keep this allowlist separate from the MMAC bridges: scalar kernels have
    different element types, while the native MMAC paths retain their tighter
    per-instruction ABI checks.
    """

    return _bridge_buffer_contracts(
        source,
        load_contracts={
            PTR_BUFFER_LOAD_I8: RAW_BUFFER_LOAD_I8,
            PTR_BUFFER_LOAD_I32: RAW_BUFFER_LOAD_I32,
            PTR_BUFFER_LOAD_F32: RAW_BUFFER_LOAD_F32,
            PTR_BUFFER_LOAD_V4F32: RAW_BUFFER_LOAD_V4F32,
            PTR_BUFFER_LOAD_V4I32: RAW_BUFFER_LOAD_V4I32,
        },
        store_contracts={
            PTR_BUFFER_STORE_I8: RAW_BUFFER_STORE_I8,
            PTR_BUFFER_STORE_I32: RAW_BUFFER_STORE_I32,
            PTR_BUFFER_STORE_F32: RAW_BUFFER_STORE_F32,
        },
        label="scalar",
    )


def bridge_gfx936_int8_mmac_for_llvm17(
    source: str,
) -> tuple[str, LLVM17Int8MmacBridgeStats]:
    """Map a legacy-layout gfx936 INT8 MMAC call to the DTK17 intrinsic.

    DTK17 exposes the native signed INT8 instruction through a three-operand
    intrinsic. The newer HCU dialect carries LIT, clamp, and LTS controls as
    three extra operands. The legacy instruction is equivalent only when all
    three controls are false, so every other combination fails closed.
    """

    source, buffer_stats = _bridge_buffer_contracts(
        source,
        load_contracts={
            PTR_BUFFER_LOAD_I16: RAW_BUFFER_LOAD_I16,
            PTR_BUFFER_LOAD_I32: RAW_BUFFER_LOAD_I32,
            PTR_BUFFER_LOAD_V4I32: RAW_BUFFER_LOAD_V4I32,
        },
        store_contracts={
            PTR_BUFFER_STORE_I32: RAW_BUFFER_STORE_I32,
            PTR_BUFFER_STORE_V4I32: RAW_BUFFER_STORE_V4I32,
            PTR_BUFFER_STORE_I8: RAW_BUFFER_STORE_I8,
            PTR_BUFFER_STORE_V4F32: RAW_BUFFER_STORE_V4F32,
        },
        label="signed INT8",
    )

    output: list[str] = []
    calls = 0
    declarations = 0

    for original_line in source.splitlines():
        call = _CALL_RE.match(original_line)
        if call is not None:
            controls = (call.group("lit"), call.group("clamp"), call.group("lts"))
            if controls != ("false", "false", "false"):
                raise LLVM17Int8MmacBridgeError(
                    "DTK17 signed INT8 MMAC only supports the legacy layout; "
                    f"got LIT/clamp/LTS={controls}"
                )

            calls += 1
            tag = call.group("result").lstrip("%")
            lhs_i64 = f"%flagtree.mmac.{tag}.lhs_i64"
            rhs_i64 = f"%flagtree.mmac.{tag}.rhs_i64"
            indent = call.group("indent")
            output.extend(
                [
                    f"{indent}{lhs_i64} = bitcast <2 x i32> {call.group('lhs')} to i64",
                    f"{indent}{rhs_i64} = bitcast <2 x i32> {call.group('rhs')} to i64",
                    (
                        f"{indent}{call.group('result')} = {call.group('tail') or ''}call "
                        f"<4 x i32> @{LEGACY_INT8_MMAC}("
                        f"i64 {lhs_i64}, i64 {rhs_i64}, "
                        f"<4 x i32> {call.group('acc')}){call.group('suffix')}"
                    ),
                ]
            )
            continue

        declaration = _DECL_RE.match(original_line)
        if declaration is not None:
            declarations += 1
            output.append(
                f"{declaration.group('indent')}declare <4 x i32> "
                f"@{LEGACY_INT8_MMAC}(i64, i64, <4 x i32>)"
                f"{declaration.group('suffix')}"
            )
            continue

        output.append(original_line)

    if calls == 0 and NEW_INT8_MMAC in source:
        raise LLVM17Int8MmacBridgeError(
            f"cannot parse signed INT8 MMAC contract: {NEW_INT8_MMAC}"
        )
    if calls and declarations != 1:
        raise LLVM17Int8MmacBridgeError(
            "expected one signed INT8 MMAC declaration when calls are present; "
            f"got {declarations}"
        )
    bridged = "\n".join(output)
    if source.endswith("\n"):
        bridged += "\n"
    forbidden = (
        NEW_INT8_MMAC,
    )
    for contract in forbidden:
        if contract in bridged:
            raise LLVM17Int8MmacBridgeError(
                f"unresolved gfx936 LLVM17 signed INT8 contract: {contract}"
            )
    return bridged, LLVM17Int8MmacBridgeStats(
        calls=calls,
        make_buffer_calls=buffer_stats.make_buffer_calls,
        raw_buffer_load_calls=buffer_stats.raw_buffer_load_calls,
        raw_buffer_store_calls=buffer_stats.raw_buffer_store_calls,
    )


_FP16_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[-A-Za-z$._0-9]+) = "
    r"(?P<tail>tail )?call <4 x float> @" + re.escape(NEW_FP16_MMAC) + r"\("
    r"<4 x half> (?P<lhs>%[-A-Za-z$._0-9]+), "
    r"<4 x half> (?P<rhs>%[-A-Za-z$._0-9]+), "
    r"<4 x float> (?P<acc>[^,]+), "
    r"i1 (?P<lit>[^,]+), i1 (?P<lts>[^)]+)\)"
    r"(?P<suffix>.*)$"
)

_FP16_DECL_RE = re.compile(
    r"^(?P<indent>\s*)declare <4 x float> @" + re.escape(NEW_FP16_MMAC)
    + r"\(<4 x half>, <4 x half>, <4 x float>, "
    r"i1(?: immarg)?, i1(?: immarg)?\)(?P<suffix>.*)$"
)


def bridge_gfx936_fp16_mmac_for_llvm17(
    source: str,
) -> tuple[str, LLVM17MmacBridgeStats]:
    """Map the gfx936 FP16 MMAC contract used by software FP8 dot.

    The DTK17 intrinsic has no LIT or LTS operands.  The bridge is equivalent
    only when both controls are false and when packed inputs use the observed
    i16-load overload. The output may retain the FP32 accumulator or use the
    explicit quantized i8 epilogue.
    """

    source, buffer_stats = _bridge_buffer_contracts(
        source,
        load_contracts={PTR_BUFFER_LOAD_I16: RAW_BUFFER_LOAD_I16},
        store_contracts={
            PTR_BUFFER_STORE_F32: RAW_BUFFER_STORE_F32,
            PTR_BUFFER_STORE_I8: RAW_BUFFER_STORE_I8,
        },
        label="FP16 MMAC",
    )
    output: list[str] = []
    calls = 0
    declarations = 0

    for original_line in source.splitlines():
        call = _FP16_CALL_RE.match(original_line)
        if call is not None:
            controls = (call.group("lit"), call.group("lts"))
            if controls != ("false", "false"):
                raise LLVM17MmacBridgeError(
                    "DTK17 FP16 MMAC only supports the legacy layout; "
                    f"got LIT/LTS={controls}"
                )
            calls += 1
            output.append(
                f"{call.group('indent')}{call.group('result')} = "
                f"{call.group('tail') or ''}call <4 x float> @{LEGACY_FP16_MMAC}("
                f"<4 x half> {call.group('lhs')}, <4 x half> {call.group('rhs')}, "
                f"<4 x float> {call.group('acc')}){call.group('suffix')}"
            )
            continue

        declaration = _FP16_DECL_RE.match(original_line)
        if declaration is not None:
            declarations += 1
            output.append(
                f"{declaration.group('indent')}declare <4 x float> "
                f"@{LEGACY_FP16_MMAC}(<4 x half>, <4 x half>, <4 x float>)"
                f"{declaration.group('suffix')}"
            )
            continue

        output.append(original_line)

    if calls == 0 and NEW_FP16_MMAC in source:
        raise LLVM17MmacBridgeError(
            f"cannot parse FP16 MMAC contract: {NEW_FP16_MMAC}"
        )
    if calls and declarations != 1:
        raise LLVM17MmacBridgeError(
            "expected one FP16 MMAC declaration when calls are present; "
            f"got {declarations}"
        )

    bridged = "\n".join(output)
    if source.endswith("\n"):
        bridged += "\n"
    if NEW_FP16_MMAC in bridged:
        raise LLVM17MmacBridgeError(
            f"unresolved gfx936 LLVM17 FP16 MMAC contract: {NEW_FP16_MMAC}"
        )
    return bridged, LLVM17MmacBridgeStats(
        calls=calls,
        make_buffer_calls=buffer_stats.make_buffer_calls,
        raw_buffer_load_calls=buffer_stats.raw_buffer_load_calls,
        raw_buffer_store_calls=buffer_stats.raw_buffer_store_calls,
    )

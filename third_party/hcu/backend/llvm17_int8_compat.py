"""Focused LLVM17 compatibility bridge for gfx936 signed INT8 dot.

The in-tree LLVM lowering emits the newer HCU MMAC contract and opaque
address-space-8 buffer descriptors.  The product DTK17 toolchain selects the
same native instruction through a legacy MMAC intrinsic and classic
``<4 x i32>`` raw-buffer descriptors.  This module bridges only the contracts
observed in a signed INT8 dot kernel and rejects every other overload.
"""

from dataclasses import dataclass
import re


class LLVM17Int8MmacBridgeError(ValueError):
    """Raised when a new MMAC contract cannot be represented by DTK17."""


@dataclass(frozen=True)
class LLVM17Int8MmacBridgeStats:
    calls: int = 0
    make_buffer_calls: int = 0
    raw_buffer_load_calls: int = 0
    raw_buffer_store_calls: int = 0


NEW_INT8_MMAC = "llvm.hcu.mmac.i32.16x16x32.i8"
LEGACY_INT8_MMAC = "llvm.amdgcn.mmac.i32.16x16x32i8"
MAKE_BUFFER_RSRC = "llvm.amdgcn.make.buffer.rsrc.p8.p1"
PTR_BUFFER_LOAD_I16 = "llvm.amdgcn.raw.ptr.buffer.load.i16"
RAW_BUFFER_LOAD_I16 = "llvm.amdgcn.raw.buffer.load.i16"
PTR_BUFFER_STORE_I32 = "llvm.amdgcn.raw.ptr.buffer.store.i32"
RAW_BUFFER_STORE_I32 = "llvm.amdgcn.raw.buffer.store.i32"

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
        raise LLVM17Int8MmacBridgeError(
            "DTK17 signed INT8 bridge cannot encode a nonzero buffer stride: "
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
        raise LLVM17Int8MmacBridgeError(
            f"cannot rewrite {contract} resource operand: {line}"
        )
    return rewritten


def _reject_unknown_buffer_overloads(source: str) -> None:
    allowed = {
        _PTR_BUFFER_LOAD_PREFIX: {PTR_BUFFER_LOAD_I16},
        _PTR_BUFFER_STORE_PREFIX: {PTR_BUFFER_STORE_I32},
    }
    for prefix, allowed_symbols in allowed.items():
        symbols = set(
            re.findall(r"@(" + re.escape(prefix) + r"[^\s(]*)\(", source)
        )
        unexpected = symbols - allowed_symbols
        if unexpected:
            raise LLVM17Int8MmacBridgeError(
                "unsupported gfx936 DTK17 signed INT8 buffer overloads: "
                + ", ".join(sorted(unexpected))
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

    _reject_unknown_buffer_overloads(source)

    output: list[str] = []
    calls = 0
    declarations = 0
    make_buffer_calls = 0
    make_buffer_declarations = 0
    raw_buffer_load_calls = 0
    raw_buffer_load_declarations = 0
    raw_buffer_store_calls = 0
    raw_buffer_store_declarations = 0

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

        if "@" + PTR_BUFFER_LOAD_I16 + "(" in original_line:
            raw_buffer_load = original_line.replace(
                "@" + PTR_BUFFER_LOAD_I16, "@" + RAW_BUFFER_LOAD_I16
            )
            if original_line.lstrip().startswith("declare "):
                raw_buffer_load_declarations += 1
            else:
                raw_buffer_load_calls += 1
            output.append(_replace_resource_type(raw_buffer_load, PTR_BUFFER_LOAD_I16))
            continue

        if "@" + PTR_BUFFER_STORE_I32 + "(" in original_line:
            raw_buffer_store = original_line.replace(
                "@" + PTR_BUFFER_STORE_I32, "@" + RAW_BUFFER_STORE_I32
            )
            if original_line.lstrip().startswith("declare "):
                raw_buffer_store_declarations += 1
            else:
                raw_buffer_store_calls += 1
            output.append(_replace_resource_type(raw_buffer_store, PTR_BUFFER_STORE_I32))
            continue

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
    if make_buffer_calls and make_buffer_declarations != 1:
        raise LLVM17Int8MmacBridgeError(
            "expected one make-buffer declaration when calls are present; "
            f"got {make_buffer_declarations}"
        )
    if raw_buffer_load_calls and raw_buffer_load_declarations != 1:
        raise LLVM17Int8MmacBridgeError(
            "expected one i16 raw-buffer load declaration when calls are present; "
            f"got {raw_buffer_load_declarations}"
        )
    if raw_buffer_store_calls and raw_buffer_store_declarations != 1:
        raise LLVM17Int8MmacBridgeError(
            "expected one i32 raw-buffer store declaration when calls are present; "
            f"got {raw_buffer_store_declarations}"
        )

    bridged = "\n".join(output)
    if source.endswith("\n"):
        bridged += "\n"
    forbidden = (
        NEW_INT8_MMAC,
        MAKE_BUFFER_RSRC,
        "llvm.amdgcn.raw.ptr.buffer",
        "ptr addrspace(8)",
    )
    for contract in forbidden:
        if contract in bridged:
            raise LLVM17Int8MmacBridgeError(
                f"unresolved gfx936 LLVM17 signed INT8 contract: {contract}"
            )
    return bridged, LLVM17Int8MmacBridgeStats(
        calls=calls,
        make_buffer_calls=make_buffer_calls,
        raw_buffer_load_calls=raw_buffer_load_calls,
        raw_buffer_store_calls=raw_buffer_store_calls,
    )

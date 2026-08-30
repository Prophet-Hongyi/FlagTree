"""Compatibility contracts between the common frontend and MThreads IR."""


def test_ir_function_exposes_frontend_finalize_hook():
    from triton._C.libtriton import ir

    assert hasattr(ir.function, "finalize")

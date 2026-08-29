from types import SimpleNamespace


def test_ppu_aabs_preserves_physical_dot_tile(monkeypatch):
    import triton.runtime.adjust_kernel_param as adjust_kernel_param

    load_map = {
        "BLOCK_M": "M",
        "BLOCK_N": "N",
        "BLOCK_K": "K",
    }
    general_dot_maps = (
        {"BLOCK_M": {"lhs"}},
        {"BLOCK_K": {"lhs", "rhs"}},
        {"BLOCK_N": {"rhs"}},
    )
    monkeypatch.setattr(adjust_kernel_param, "FLAGTREE_BACKEND", "ppu")
    monkeypatch.setattr(
        adjust_kernel_param,
        "analyze_kernel_dependencies",
        lambda _fn, pre_hook_fn=None: (
            load_map,
            {},
            {},
            {},
            *general_dot_maps,
        ),
    )

    config = SimpleNamespace(
        kwargs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
        pre_hook=None,
    )
    current = dict(config.kwargs)
    adjust_kernel_param.auto_adjust_block_sizes(
        {"M": 8, "N": 8, "K": 8},
        object(),
        [config],
        current,
        config,
    )

    assert current == {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}
    assert config.kwargs == current

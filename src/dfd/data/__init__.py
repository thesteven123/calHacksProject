_PAIRED = ("CompositeDataset", "DeepFakeFaceDataset", "SowaibaDataset", "verify_no_group_leakage")
__all__ = list(_PAIRED)


def __getattr__(name: str):
    # Lazy: torch-free submodules (tar_index, wilddeepfake_index) must import on the CPU-only Modal ingest image.
    if name in _PAIRED:
        from dfd.data import paired

        return getattr(paired, name)
    raise AttributeError(f"module 'dfd.data' has no attribute {name!r}")

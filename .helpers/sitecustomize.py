"""sitecustomize: stub torch_sparse so LLaGA processed_data.pt can unpickle.

The LLaGA-distributed ``processed_data.pt`` for ogbn-arxiv stores
``data.adj_t`` as ``torch_sparse.SparseTensor``. The training pipeline only
needs ``edge_index`` from this object, but ``torch.load`` still needs to
resolve the ``SparseTensor`` class during unpickling. Installing the native
``torch_sparse`` extension on this cluster is fragile (CUDA 12.8 / Torch
2.11), so we register tiny stub modules under ``sys.modules`` before any
training code runs. This file is auto-imported by Python at interpreter
startup when its containing directory is on ``PYTHONPATH``.

This is a runtime shim only -- no training-pipeline source file is modified.
"""

from __future__ import annotations

import sys
import types


def _install_torch_sparse_stub() -> None:
    if "torch_sparse" in sys.modules:
        return

    ts_mod = types.ModuleType("torch_sparse")
    ts_mod.__path__ = []
    storage_mod = types.ModuleType("torch_sparse.storage")
    tensor_mod = types.ModuleType("torch_sparse.tensor")

    class _Stub:
        def __setstate__(self, s):
            if isinstance(s, dict):
                for k, v in s.items():
                    setattr(self, k, v)

    storage_mod.SparseStorage = _Stub
    tensor_mod.SparseTensor = _Stub
    ts_mod.SparseTensor = _Stub
    ts_mod.SparseStorage = _Stub
    sys.modules["torch_sparse"] = ts_mod
    sys.modules["torch_sparse.storage"] = storage_mod
    sys.modules["torch_sparse.tensor"] = tensor_mod


_install_torch_sparse_stub()

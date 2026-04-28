"""Per-dataset loaders.

Each module exposes a ``load(config, split, seed) -> (node_data, adj, class_names, split_ids)``.
``LOADERS`` is the dataset-name -> loader registry consumed by ``graph.load_tag_dataset``.
"""

from . import cora, ogbn_arxiv, ogbn_products, pubmed
from ._common import load_llaga_processed_data

LOADERS = {
    "cora": cora.load,
    "pubmed": pubmed.load,
    "ogbn-arxiv": ogbn_arxiv.load,
    "ogbn-products": ogbn_products.load,
}

__all__ = ["LOADERS", "load_llaga_processed_data"]

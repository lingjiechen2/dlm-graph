"""Sync LLaGA datasets from the official Box share into a fixed local folder.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/sync_llaga_datasets.py --list-only

Examples:
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/sync_llaga_datasets.py --datasets cora,pubmed --list-only
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/sync_llaga_datasets.py --datasets cora,pubmed --download
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/sync_llaga_datasets.py --datasets ogbn-arxiv --download --include-top-level
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


SHARED_URL = "https://utexas.app.box.com/s/i7y03rzm40xt9bjbaj0dfdgxeyjx77gb"
SHARED_NAME = "i7y03rzm40xt9bjbaj0dfdgxeyjx77gb"
REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
DEFAULT_OUT_ROOT = REPO_ROOT / ".datasets" / "llaga"
MANIFEST_PATH = DEFAULT_OUT_ROOT / "manifest.json"

TOP_LEVEL_DATASETS = ("cora", "pubmed", "ogbn-arxiv", "ogbn-products")
CORE_FILE_PATTERNS = (
    re.compile(r"^processed_data(_link_notest)?\.pt$"),
    re.compile(r"^sampled_2_10_(train|val|test)\.jsonl$"),
    re.compile(r"^edge_sampled_2_10_only_(train|test)\.jsonl$"),
)
CORE_TOP_LEVEL_PATTERNS = (
    re.compile(r"^laplacian_2_10\.pt$"),
    re.compile(r"^laplacian_2_20\.pt$"),
    re.compile(r"^laplacian_2_5\.pt$"),
)


@dataclass(frozen=True)
class RemoteItem:
    name: str
    item_id: int
    item_type: str
    size: int
    parent_folder_id: int | None = None

    @property
    def typed_id(self) -> str:
        prefix = "f" if self.item_type == "file" else "d"
        return f"{prefix}_{self.item_id}"


def _extract_poststream_data(html: str) -> dict:
    match = re.search(r"Box\.postStreamData = (\{.*?\});</script>", html)
    if not match:
        raise RuntimeError("Failed to locate Box.postStreamData in shared page HTML.")
    return json.loads(match.group(1))


def _fetch_folder_html(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def fetch_root_listing() -> tuple[dict[str, RemoteItem], list[RemoteItem]]:
    html = _fetch_folder_html(SHARED_URL)
    data = _extract_poststream_data(html)
    items = data["/app-api/enduserapp/shared-folder"]["items"]
    datasets: dict[str, RemoteItem] = {}
    top_level_files: list[RemoteItem] = []
    for item in items:
        remote = RemoteItem(
            name=item["name"],
            item_id=int(item["id"]),
            item_type=item["type"],
            size=int(item.get("itemSize") or 0),
            parent_folder_id=int(item["parentFolderID"]) if item.get("parentFolderID") else None,
        )
        if remote.item_type == "folder" and remote.name in TOP_LEVEL_DATASETS:
            datasets[remote.name] = remote
        elif remote.item_type == "file":
            top_level_files.append(remote)
    return datasets, top_level_files


def _parse_folder_page(folder_url: str) -> tuple[list[dict], int]:
    html = _fetch_folder_html(folder_url)
    data = _extract_poststream_data(html)
    folder_payload = data["/app-api/enduserapp/shared-folder"]
    return folder_payload["items"], int(folder_payload.get("pageCount", 1))


def fetch_folder_listing(folder: RemoteItem) -> list[RemoteItem]:
    first_page_url = f"{SHARED_URL}/folder/{folder.item_id}"
    items, page_count = _parse_folder_page(first_page_url)
    all_items = list(items)
    for page in range(2, page_count + 1):
        page_items, _ = _parse_folder_page(f"{first_page_url}?page={page}")
        all_items.extend(page_items)
    result: list[RemoteItem] = []
    for item in all_items:
        result.append(
            RemoteItem(
                name=item["name"],
                item_id=int(item["id"]),
                item_type=item["type"],
                size=int(item.get("itemSize") or 0),
                parent_folder_id=int(item["parentFolderID"]) if item.get("parentFolderID") else None,
            )
        )
    return result


def build_manifest(datasets: Iterable[str]) -> dict:
    root_dirs, top_level_files = fetch_root_listing()
    manifest: dict[str, object] = {
        "shared_url": SHARED_URL,
        "shared_name": SHARED_NAME,
        "datasets": {},
        "top_level_files": [
            {
                "name": item.name,
                "id": item.item_id,
                "type": item.item_type,
                "size": item.size,
            }
            for item in top_level_files
        ],
    }
    for name in datasets:
        if name not in root_dirs:
            raise KeyError(f"Dataset {name!r} not found in LLaGA root share.")
        folder = root_dirs[name]
        children = fetch_folder_listing(folder)
        manifest["datasets"][name] = {
            "folder_id": folder.item_id,
            "size": folder.size,
            "files": [
                {
                    "name": item.name,
                    "id": item.item_id,
                    "type": item.item_type,
                    "size": item.size,
                }
                for item in children
            ],
        }
    return manifest


def download_file(remote_file_id: int, destination: Path) -> None:
    url = (
        "https://utexas.app.box.com/index.php"
        f"?rm=box_download_shared_file&shared_name={SHARED_NAME}&file_id=f_{remote_file_id}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def print_manifest_summary(manifest: dict) -> None:
    print(f"Shared URL: {manifest['shared_url']}")
    for ds_name, ds_meta in manifest["datasets"].items():
        files = ds_meta["files"]
        total_size = sum(int(f["size"]) for f in files if f["type"] == "file")
        print(f"[dataset] {ds_name}: {len(files)} entries, {total_size / (1024**3):.2f} GiB")
        for entry in files:
            print(
                f"  - {entry['name']}  ({entry['type']}, {int(entry['size']) / (1024**2):.2f} MiB)"
            )
    if manifest["top_level_files"]:
        print("[top-level files]")
        for entry in manifest["top_level_files"]:
            print(
                f"  - {entry['name']}  ({entry['type']}, {int(entry['size']) / (1024**2):.2f} MiB)"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="cora,pubmed,ogbn-arxiv",
        help="Comma-separated subset of cora,pubmed,ogbn-arxiv,ogbn-products.",
    )
    parser.add_argument(
        "--out_root",
        default=str(DEFAULT_OUT_ROOT),
        help="Local directory to store aligned LLaGA datasets.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only fetch and save the remote manifest; do not download files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all files under the selected dataset folders.",
    )
    parser.add_argument(
        "--profile",
        default="core",
        choices=["core", "full"],
        help="core: only processed_data + sampled jsonl + edge_sampled jsonl; full: every file.",
    )
    parser.add_argument(
        "--include-top-level",
        action="store_true",
        help="Also download top-level files such as laplacian_2_*.pt into out_root.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally with the same byte size.",
    )
    return parser.parse_args()


def should_download(entry_name: str, profile: str, *, top_level: bool = False) -> bool:
    if profile == "full":
        return True
    patterns = CORE_TOP_LEVEL_PATTERNS if top_level else CORE_FILE_PATTERNS
    return any(p.match(entry_name) for p in patterns)


def main() -> None:
    args = parse_args()
    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(selected)
    save_manifest(manifest, MANIFEST_PATH)
    print_manifest_summary(manifest)
    print(f"[manifest] {MANIFEST_PATH}")

    if args.list_only or not args.download:
        return

    for ds_name in selected:
        ds_meta = manifest["datasets"][ds_name]
        local_dir = out_root / ds_name
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in ds_meta["files"]:
            if entry["type"] != "file":
                continue
            if not should_download(entry["name"], args.profile, top_level=False):
                continue
            dst = local_dir / entry["name"]
            if (
                args.skip_existing
                and dst.exists()
                and dst.stat().st_size == int(entry["size"])
            ):
                print(f"[skip] {dst}")
                continue
            print(f"[download] {dst}")
            download_file(int(entry["id"]), dst)

    if args.include_top_level:
        for entry in manifest["top_level_files"]:
            if entry["type"] != "file":
                continue
            if not should_download(entry["name"], args.profile, top_level=True):
                continue
            dst = out_root / entry["name"]
            if (
                args.skip_existing
                and dst.exists()
                and dst.stat().st_size == int(entry["size"])
            ):
                print(f"[skip] {dst}")
                continue
            print(f"[download] {dst}")
            download_file(int(entry["id"]), dst)


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from .utils import flatten_nested_modal_inputs, normalize_text

logger = logging.getLogger(__name__)

VISDOC_TASK_NAME = "visdoc"
VISDOC_QUERY_INSTRUCTION = "Find a document image that matches the given query."


class VisDocDataset(TorchDataset):
    """Local VisDoc/ColPali query-page training pairs.

    The local dataset is a Hugging Face parquet dataset with one text query and
    one positive document page image per row. It intentionally has no explicit
    hard negatives; the trainer uses in-batch negatives when
    num_hard_negatives=0.
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        val_ratio: float,
        seed: int,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        self.data_path = data_path
        self.split = split
        self.seed = seed
        self.val_ratio = val_ratio

        source = self._load_split(data_path, split=split, val_ratio=val_ratio, seed=seed, cache_dir=cache_dir)
        if max_samples is not None and len(source) > max_samples:
            source = source.select(range(max_samples))
        self.dataset = source

        logger.info(
            "Loaded VisDoc split=%s examples=%d data_path=%s",
            split,
            len(self.dataset),
            data_path,
        )

    @staticmethod
    def _load_split(
        data_path: str,
        split: str,
        val_ratio: float,
        seed: int,
        cache_dir: Optional[str],
    ) -> Dataset:
        if val_ratio > 0.0:
            train_data = load_dataset(data_path, split="train", cache_dir=cache_dir)
            split_dict = train_data.train_test_split(test_size=val_ratio, seed=seed)
            return split_dict["train"] if split == "train" else split_dict["test"]

        if split == "train":
            return load_dataset(data_path, split="train", cache_dir=cache_dir)

        try:
            return load_dataset(data_path, split="test", cache_dir=cache_dir)
        except Exception:
            train_data = load_dataset(data_path, split="train", cache_dir=cache_dir)
            return train_data.select([])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = dict(self.dataset[int(index)])
        row["task"] = VISDOC_TASK_NAME
        row["sample_index"] = int(index)
        return row

    @property
    def task_slices(self) -> List[tuple[str, int, int]]:
        return [(VISDOC_TASK_NAME, 0, len(self.dataset))]


class TaskAwareBatchSampler:
    def __init__(
        self,
        task_slices: Sequence[tuple[str, int, int]],
        batch_size: int,
        drop_last: bool,
        seed: int,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        self.task_slices = list(task_slices)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        all_batches: List[List[int]] = []
        for _, offset, size in self.task_slices:
            if size == 0:
                continue
            local_indices = torch.randperm(size, generator=generator).tolist()
            global_indices = [offset + idx for idx in local_indices]
            if self.drop_last:
                usable = (len(global_indices) // self.batch_size) * self.batch_size
                global_indices = global_indices[:usable]
            for start in range(0, len(global_indices), self.batch_size):
                batch = global_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    all_batches.append(batch)

        if all_batches:
            order = torch.randperm(len(all_batches), generator=generator).tolist()
            all_batches = [all_batches[i] for i in order]

        if self.num_replicas > 1 and all_batches:
            remainder = len(all_batches) % self.num_replicas
            if remainder != 0:
                if self.drop_last:
                    all_batches = all_batches[: len(all_batches) - remainder]
                else:
                    all_batches.extend(all_batches[: self.num_replicas - remainder])

        for batch in all_batches[self.rank :: self.num_replicas]:
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for _, _, size in self.task_slices:
            if self.drop_last:
                total_batches += size // self.batch_size
            else:
                total_batches += math.ceil(size / self.batch_size)
        if self.drop_last:
            return total_batches // self.num_replicas
        return math.ceil(total_batches / self.num_replicas)


class VisDocPairCollator:
    def __init__(
        self,
        num_hard_negatives: int = 0,
        query_instruction: str = VISDOC_QUERY_INSTRUCTION,
        image_cache_dir: Optional[str] = None,
    ) -> None:
        if num_hard_negatives != 0:
            raise ValueError("VisDoc training data has no hard negatives; use num_hard_negatives=0.")
        self.num_hard_negatives = num_hard_negatives
        self.query_instruction = query_instruction
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        if self.image_cache_dir is not None:
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _decode_image_dict(image_dict: Dict[str, Any]) -> Optional[Any]:
        image_bytes = image_dict.get("bytes")
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_path = image_dict.get("path")
        if image_path:
            path = Path(str(image_path))
            if path.exists():
                return str(path)
        return None

    @staticmethod
    def _safe_filename(value: Any, fallback: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            text = fallback
        text = text.replace("\\", "/").split("/")[-1]
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
        return text[:160] or fallback

    def _cache_path(self, row: Dict[str, Any]) -> Path:
        assert self.image_cache_dir is not None
        source = self._safe_filename(row.get("source"), "unknown_source")
        sample_index = int(row.get("sample_index", -1))
        image_name = self._safe_filename(row.get("image_filename"), f"sample_{sample_index}")
        return self.image_cache_dir / source / f"{sample_index:08d}_{image_name}.png"

    @staticmethod
    def _atomic_save_image(image: Any, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_name(f"{dest_path.name}.tmp.{os.getpid()}")
        image.save(tmp_path, format="PNG")
        os.replace(tmp_path, dest_path)

    def _materialize_image(self, image: Any, row: Dict[str, Any]) -> Any:
        if self.image_cache_dir is None or isinstance(image, str):
            return image
        if not callable(getattr(image, "save", None)):
            return image
        dest_path = self._cache_path(row)
        if not dest_path.exists():
            self._atomic_save_image(image, dest_path)
        return str(dest_path)

    def _coerce_image(self, image: Any, row: Dict[str, Any]) -> Optional[Any]:
        if image is None:
            return None
        if isinstance(image, dict):
            image = self._decode_image_dict(image)
        if image is None:
            return None
        return self._materialize_image(image, row)

    @staticmethod
    def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_filename": row.get("image_filename"),
            "source": row.get("source"),
            "page": row.get("page"),
            "answer": row.get("answer"),
            "options": row.get("options"),
            "model": row.get("model"),
            "answer_type": row.get("answer_type"),
            "sample_index": row.get("sample_index"),
        }

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not examples:
            return {
                "task": VISDOC_TASK_NAME,
                "queries": [],
                "positives": [],
                "negatives": [],
                "flat_negatives": [],
                "sample_indices": [],
                "metadata": [],
            }

        queries: List[Dict[str, Optional[Any]]] = []
        positives: List[Dict[str, Optional[Any]]] = []
        negatives: List[List[Dict[str, Optional[Any]]]] = []
        sample_indices: List[int] = []
        metadata: List[Dict[str, Any]] = []

        for example in examples:
            query_text = normalize_text(example.get("query"))
            page_image = self._coerce_image(example.get("image"), example)
            if not query_text or page_image is None:
                continue

            queries.append({
                "text": query_text,
                "image": None,
                "raw_image_path": None,
                "instruction": self.query_instruction,
            })
            positives.append({
                "text": None,
                "image": page_image,
                "raw_image_path": example.get("image_filename"),
                "instruction": None,
            })
            negatives.append([])
            sample_indices.append(int(example.get("sample_index", -1)))
            metadata.append(self._metadata(example))

        return {
            "task": VISDOC_TASK_NAME,
            "queries": queries,
            "positives": positives,
            "negatives": negatives,
            "flat_negatives": flatten_nested_modal_inputs(negatives),
            "sample_indices": sample_indices,
            "metadata": metadata,
        }

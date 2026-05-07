from __future__ import annotations

import ast
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

QWEN_MODALITY_TOKEN_RE = re.compile(r"<\|(image|video)_\d+\|>")
GENERIC_MODALITY_TOKEN_RE = re.compile(r"<(image|video)>", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def strip_qwen_modality_placeholders(text: str) -> str:
    text = QWEN_MODALITY_TOKEN_RE.sub(" ", text)
    text = GENERIC_MODALITY_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def normalize_text(text: Any, strip_placeholders: bool = True) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    text = str(text).strip()
    if not text:
        return ""
    if strip_placeholders:
        text = strip_qwen_modality_placeholders(text)
    return text.strip()



def parse_maybe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return ["" if item is None else str(item) for item in value]
    if isinstance(value, tuple):
        return ["" if item is None else str(item) for item in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                return [raw]
            if isinstance(parsed, (list, tuple)):
                return ["" if item is None else str(item) for item in parsed]
        return [raw]
    return [str(value)]



def resolve_image_path(path: Any, image_root: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path_str = str(path).strip()
    if not path_str:
        return None
    if path_str.startswith(("http://", "https://", "file://")):
        return path_str
    candidate = Path(path_str)
    if candidate.is_absolute():
        return str(candidate) if candidate.exists() else None
    if image_root:
        image_root_path = Path(image_root)
        candidates = [image_root_path / candidate]
        if candidate.parts and candidate.parts[0] == "images":
            candidates.append(image_root_path.parent / candidate)
        for joined in candidates:
            if joined.exists():
                return str(joined)
        return None
    return path_str



def ensure_file_uri(path: str) -> str:
    if path.startswith(("http://", "https://", "file://")):
        return path
    return "file://" + os.path.abspath(path)



def pair_text_and_images(texts: Sequence[str], images: Sequence[str]) -> List[Dict[str, Optional[str]]]:
    size = max(len(texts), len(images))
    pairs: List[Dict[str, Optional[str]]] = []
    for idx in range(size):
        text = texts[idx] if idx < len(texts) else ""
        image = images[idx] if idx < len(images) else ""
        pairs.append({"text": text or None, "image": image or None})
    return pairs



def flatten_nested_modal_inputs(batch: Sequence[Sequence[Dict[str, Optional[str]]]]) -> List[Dict[str, Optional[str]]]:
    flat: List[Dict[str, Optional[str]]] = []
    for group in batch:
        flat.extend(group)
    return flat

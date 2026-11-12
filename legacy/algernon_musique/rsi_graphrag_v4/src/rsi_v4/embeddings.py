"""Local NV-Embed-v2 encoder. GPU is used only when a device has enough free memory."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


NV_PATH = (
    "/data2-HDD-SATA-20T/nzq/huggingface_model/hub/models--nvidia--NV-Embed-v2/"
    "snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef"
)


class NVEmbedEncoder:
    name = "nvidia/NV-Embed-v2"

    def __init__(self, model_path: str = NV_PATH, batch_size: int = 2, max_length: int = 512) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.device = "cpu"
        self.load_error = ""
        self.cache = {}

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel

        device = "cpu"
        dtype = torch.float32
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(index)
                if free > 12 * (1024 ** 3):
                    device = f"cuda:{index}"
                    dtype = torch.float16
                    break
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        config._name_or_path = self.model_path
        if getattr(config, "text_config", None) is not None:
            config.text_config._name_or_path = self.model_path
        self.model = AutoModel.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: Sequence[str], instruction: str = "") -> np.ndarray:
        self.load()
        import torch

        wrapped = f"Instruct: {instruction}\nQuery: " if instruction else ""
        texts = list(texts)
        missing = [text for text in texts if (instruction, text) not in self.cache]
        batch_size = self.batch_size
        index = 0
        while index < len(missing):
            current = missing[index : index + batch_size]
            try:
                with torch.no_grad():
                    result = self.model.encode(
                        prompts=current,
                        instruction=wrapped,
                        max_length=self.max_length,
                    )
                if isinstance(result, torch.Tensor):
                    result = result.detach().cpu().numpy()
                result = np.asarray(result)
                for text, row in zip(current, result):
                    self.cache[(instruction, text)] = row
                index += len(current)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    continue
                raise
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self.cache[(instruction, text)] for text in texts])

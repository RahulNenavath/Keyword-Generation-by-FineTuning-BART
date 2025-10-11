import torch
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
from schema import DataConfig, PromptConfig
from transformers import PreTrainedTokenizerBase
from datasets import DatasetDict, load_dataset


class KeywordDataModule:
    def __init__(self, data_cfg: DataConfig, prompt_cfg: PromptConfig):
        self.cfg = data_cfg
        self.prompt = prompt_cfg

    # --- helpers ---
    def _to_keyword_list(self, v: Any) -> List[str]:
        """
        Normalize 'keywords' field to a list[str].
        Accepts list[str] or string with separators (\n ; ,).
        """
        if v is None:
            return []
        if isinstance(v, list):
            items = [str(x).strip() for x in v if isinstance(x, (str, int, float))]
        else:
            s = str(v)
            # split on newline/semicolon/commas
            parts = []
            for chunk in s.split("\n"):
                for p in chunk.split(";"):
                    parts.extend(p.split(","))
            items = [p.strip() for p in parts]
        # drop empties & deduplicate case-insensitively (preserve order)
        seen, out = set(), []
        for k in items:
            if not k:
                continue
            kn = k.casefold()
            if kn not in seen:
                seen.add(kn)
                out.append(k)
        return out

    def _format_example(self, ex: Dict[str, Any]) -> Dict[str, str]:
        doc = (ex.get(self.cfg.text_field, "") or "").strip()
        kw_list = self._to_keyword_list(ex.get(self.cfg.keywords_field, []))
        labels = self.prompt.sep.join(kw_list)  # e.g., "; ".join(...)
        prompt = (
            f"{self.prompt.system_preamble}\n\n"
            f"DOCUMENT:\n{doc}\n\n"
            f"{self.prompt.response_tag}"
        )
        return {"text": prompt, "labels": labels}

    def load(self, dsd: Optional[DatasetDict] = None) -> DatasetDict:
        # Load DatasetDict (local saved or hub)
        if dsd is None:
            if self.cfg.hf_path_or_none is None:
                raise ValueError("Provide a DatasetDict or set DataConfig.hf_path_or_none.")
            try:
                dsd = DatasetDict.load_from_disk(self.cfg.hf_path_or_none)
            except Exception:
                # assuming a hub dataset id that exposes splits
                dsd = load_dataset(self.cfg.hf_path_or_none)

        mapped = DatasetDict()
        for split in dsd.keys():
            ds = dsd[split]

            # Optional capping for speed/memory
            if split == "train" and self.cfg.max_train_samples:
                ds = ds.select(range(min(self.cfg.max_train_samples, len(ds))))
            if split in ("validation", "test") and self.cfg.max_eval_samples:
                ds = ds.select(range(min(self.cfg.max_eval_samples, len(ds))))

            # Map to {"text","labels"}
            ds = ds.map(
                self._format_example,
                remove_columns=[c for c in ds.column_names if c not in ("text", "labels")],
            )

            # Filter out rows with empty labels (no keywords)
            ds = ds.filter(lambda ex: bool(ex["labels"] and ex["labels"].strip()))

            mapped[split] = ds

        # Quick sanity: ensure expected columns exist
        for s in mapped.keys():
            cols = set(mapped[s].column_names)
            if not {"text", "labels"}.issubset(cols):
                raise RuntimeError(f"Split '{s}' must contain 'text' and 'labels', got: {cols}")

        return mapped
    

@dataclass
class KeywordCompletionCollator:
    tokenizer: PreTrainedTokenizerBase
    response_template: str = "KEYWORDS:"
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8
    add_eos: bool = True

    def _get_text_labels(self, ex: Dict[str, Any]) -> Tuple[str, str]:
        if "text" in ex and "labels" in ex:
            return ex["text"], ex["labels"]
        if "prompt" in ex and "response" in ex:
            return ex["prompt"], ex["response"]
        raise KeyError(f"Example must contain ('text','labels') or ('prompt','response'), got: {list(ex.keys())}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Normalize to (prompt, target) strings
        prompts, targets = [], []
        for ex in batch:
            p, t = self._get_text_labels(ex)
            prompts.append(p)
            targets.append("" if t is None else t)

        # Build full training strings: prompt + target (+ eos)
        eos = self.tokenizer.eos_token if (self.add_eos and self.tokenizer.eos_token) else ""
        full_texts = [
            p + (" " if (t and not p.endswith(" ")) else "") + t + eos
            for p, t in zip(prompts, targets)
        ]

        # Tokenize full strings
        enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Tokenize "prompt + response_template" to find the boundary
        tagged_prompts = [
            p if p.strip().endswith(self.response_template)
            else (p.rstrip() + " " + self.response_template)
            for p in prompts
        ]
        tag_enc = self.tokenizer(
            tagged_prompts,
            padding=True,
            truncation=True,
            max_length=input_ids.size(1),
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        tag_lens = tag_enc["attention_mask"].sum(dim=1)  # prompt+tag token lengths
        seq_lens = attention_mask.sum(dim=1)

        # Build masked labels: -100 before target start, ids on target region
        labels = input_ids.clone()
        labels[:] = -100
        for i in range(input_ids.size(0)):
            start = int(tag_lens[i].item())
            end = int(seq_lens[i].item())
            if start < end:
                labels[i, start:end] = input_ids[i, start:end]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
import torch
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from schema import Seq2SeqKwPreprocConfig
from datasets import DatasetDict
from transformers import DataCollator, AutoTokenizer, PreTrainedTokenizerBase

class Seq2SeqKeywordPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: Seq2SeqKwPreprocConfig):
        self.tok = tokenizer
        self.cfg = cfg

        # T5/FLAN-T5 specifics
        # - padding token is 0, eos is 1 for T5 tokenizer (already true in most checkpoints)
        # - pad on right for seq2seq batching
        self.tok.padding_side = "right"
        if self.tok.pad_token_id is None:
            # For T5, pad_token_id should be 0; keep fallback to eos if missing
            self.tok.pad_token_id = getattr(self.tok, "eos_token_id", 0)

    def format_fn(self, ex: Dict[str, Any]) -> Dict[str, str]:
        """Map a raw example -> {'input_text','target_text'} for seq2seq."""
        doc = (ex.get("text") or "").strip()
        kws = ex.get("keywords") or []
        # ensure keywords is list[str]
        if isinstance(kws, str):
            # tolerate accidental single string
            kws = [k.strip() for k in kws.split(";") if k.strip()]
        else:
            kws = [str(k).strip() for k in kws if str(k).strip()]

        input_text = f"{self.cfg.prefix}{doc}"
        target_text = self.cfg.sep.join(kws)

        return {"input_text": input_text, "target_text": target_text}

    def tokenize_batch(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Tokenize in SAMSum/T5 style:
          - encode inputs (source)
          - encode targets (labels), then replace pad_id -> -100 for loss ignoring
        """
        inputs = batch["input_text"]
        targets = batch["target_text"]

        # Encoder side
        model_inputs = self.tok(
            inputs,
            max_length=self.cfg.max_source_len,
            truncation=self.cfg.truncate_long_docs,
            padding=False,               # let HF collator do dynamic padding
        )

        # Decoder side (labels)
        with self.tok.as_target_tokenizer():  # T5-compatible API
            labels = self.tok(
                targets,
                max_length=self.cfg.max_target_len,
                truncation=True,
                padding=False,
            )["input_ids"]

        # Optionally append EOS to labels if tokenizer didn’t
        if self.cfg.add_eos and self.tok.eos_token_id is not None:
            eos = self.tok.eos_token_id
            labels = [seq + ([eos] if (len(seq) == 0 or seq[-1] != eos) else []) for seq in labels]

        # Replace padding in labels with -100 (trainer ignores those)
        pad_id = self.tok.pad_token_id
        labels = [[(tid if tid != pad_id else -100) for tid in seq] for seq in labels]

        model_inputs["labels"] = labels
        return model_inputs

    def apply(self, dsd: DatasetDict, num_proc: Optional[int] = None) -> DatasetDict:
        """
        Returns a new DatasetDict with columns:
          input_ids, attention_mask, labels
        """
        # 1) map to input/target strings
        dsd_fmt = dsd.map(self.format_fn, remove_columns=[
            c for split in dsd for c in dsd[split].column_names
        ] if len(dsd) and set(next(iter(dsd.values())).column_names) >= {"text", "keywords"} else None)

        # 2) tokenize
        dsd_tok = dsd_fmt.map(
            self.tokenize_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=["input_text", "target_text"],
            desc="Tokenizing (seq2seq SAMSum style)",
        )
        return dsd_tok
    
@dataclass
class DataCollatorForSeq2SeqSimple(DataCollator):
    """
    Minimal seq2seq collator:
      - dynamically pads input_ids & attention_mask
      - dynamically pads labels, then turns pad-> -100 for loss masking
    Works directly on pre-tokenized batches (SAMSum-style).
    """
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels so we can pad them independently
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        # Pad encoder inputs (dynamic)
        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels to the max length in this batch (manual)
        max_len = max(len(l) for l in labels) if labels else 0
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_labels = []
        for l in labels:
            pad_len = max_len - len(l)
            if pad_len > 0:
                l = l + [pad_id] * pad_len
            padded_labels.append(l)

        labels_t = torch.tensor(padded_labels, dtype=torch.long)

        # Replace pad tokens in labels with -100 so they’re ignored by loss
        labels_t[labels_t == pad_id] = self.label_pad_token_id

        batch["labels"] = labels_t
        return batch
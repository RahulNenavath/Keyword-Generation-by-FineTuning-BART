import re
import os
import pandas as pd
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSeq2SeqLM, PreTrainedModel
from typing import List, Dict, Union, Optional, Iterable
from datasets import Dataset, DatasetDict, Features, Value, Sequence


class CheckpointSaver:
    """
    Utility to extract the best (or last) Trainer checkpoint and save
    a clean, portable model directory with tokenizer and generation config.
    """

    @staticmethod
    def save_best_model_dir(
        trainer,
        out_dir: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        make_generation_config: bool = True,
    ) -> str:
        """
        Args:
            trainer: a Seq2SeqTrainer after `trainer.train()`
            out_dir: path to save the consolidated model folder
            tokenizer: optional tokenizer to save; if None, use trainer.tokenizer
            make_generation_config: if True, writes generation_config.json if missing

        Returns:
            out_dir (for convenience)
        """
        os.makedirs(out_dir, exist_ok=True)

        # Prefer the best checkpoint; fall back to last trainer.args.output_dir
        best = getattr(trainer.state, "best_model_checkpoint", None)
        if best is None:
            # Look for latest checkpoint subdir
            checkpoints = [
                os.path.join(trainer.args.output_dir, d)
                for d in os.listdir(trainer.args.output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(trainer.args.output_dir, d))
            ]
            if not checkpoints:
                raise FileNotFoundError(
                    "No checkpoint-* directories found. Ensure you trained with save_steps>0."
                )
            best = sorted(checkpoints, key=lambda p: int(p.split("-")[-1]))[-1]

        # Load the model weights from that checkpoint
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(best)
        model.save_pretrained(out_dir)

        # Save tokenizer
        tok = tokenizer
        if tok is None:
            raise ValueError("Tokenizer not available. Pass it explicitly or ensure trainer.tokenizer is set.")
        tok.save_pretrained(out_dir)

        # Optionally write/refresh generation_config.json
        if make_generation_config:
            try:
                model.generation_config.save_pretrained(out_dir)
            except Exception:
                pass  # not fatal

        print(f"[CheckpointSaver] Saved consolidated model to: {out_dir}")
        return out_dir


def normalize_kw_string(s: str) -> List[str]:
    if s is None:
        return []
    raw = []
    for chunk in s.split("\n"):
        raw.extend([p for w in chunk.split(";") for p in w.split(",")])
    norm, seen = [], set()
    for k in raw:
        k2 = k.strip()
        kn = k2.casefold()
        if k2 and kn not in seen:
            seen.add(kn)
            norm.append(k2)
    return norm


def f1_keywords(preds: List[List[str]], refs: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for p, r in zip(preds, refs):
        ps = set(x.casefold() for x in p)
        rs = set(x.casefold() for x in r)
        tp += len(ps & rs); fp += len(ps - rs); fn += len(rs - ps)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _clean_single(t: str) -> str:
    if not isinstance(t, str):
        return t
    # Remove ASCII control chars except '\n' (0x0A)
    t = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F]', '', t)
    # Remove CR and TAB explicitly; keep '\n'
    t = t.replace('\r', '').replace('\t', ' ')
    # Collapse spaces (but not '\n')
    t = re.sub(r'[ ]{2,}', ' ', t)
    # Trim per line
    t = '\n'.join(line.strip() for line in t.split('\n'))
    return t.strip()


def clean_key_phrase(x: Union[str, List[str]]) -> Union[str, List[str]]:
    """Clean a string or list[str] of keywords; preserves '\n' in strings."""
    if isinstance(x, list):
        return [_clean_single(t) for t in x if isinstance(t, str) and t.strip()]
    return _clean_single(x)


def _dedup_preserve_order(items: List[str]) -> List[str]:
    """Case-insensitive, order-preserving dedup (keeps first-seen casing)."""
    seen = set()
    out = []
    for it in items:
        if not isinstance(it, str):
            continue
        k = it.strip()
        if not k:
            continue
        kn = k.casefold()  # Unicode-safe
        if kn not in seen:
            seen.add(kn)
            out.append(k)
    return out

def build_hf_dataset_from_pandas(df: pd.DataFrame, seed: int = 42) -> DatasetDict:

    df = df.copy()
    # Clean text (optional; comment out next line if you don't want it)
    df["text"] = df["text"].astype(str).map(_clean_single)

    def to_keywords(v) -> List[str]:
        # split to list if string; already list => keep
        kws = v.split("\n") if isinstance(v, str) else (v if isinstance(v, list) else [])
        kws = clean_key_phrase(kws) or []
        kws = [k for k in kws if k]  # drop empties
        kws = _dedup_preserve_order(kws)
        return kws

    df["keywords"] = df["key"].apply(to_keywords)
    df = df[df["keywords"].map(len) > 0].reset_index(drop=True)

    features = Features({
        "text": Value("string"),
        "keywords": Sequence(Value("string")),
    })

    ds_all = Dataset.from_pandas(
        df[["text", "keywords"]], preserve_index=False, features=features
    ).shuffle(seed=seed)

    split = ds_all.train_test_split(test_size=0.2, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)

    return DatasetDict({
        "train": split["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })
    

def _clean_kw(s: str) -> str:
    # normalize whitespace, strip quotes/punct at ends
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'^[\'"“”‘’\(\)\[\]\{\}\-–—,:;]+|[\'"“”‘’\(\)\[\]\{\}\-–—,:;]+$', "", s)
    return s.strip()


def _postprocess(
    kws: Iterable[str],
    *,
    lowercase: bool = False,
    min_len: int = 2,
    max_len: Optional[int] = None,
    dedupe: bool = True,
) -> List[str]:
    out: List[str] = []
    seen = set()
    for k in kws:
        k2 = _clean_kw(k)
        if lowercase:
            k2 = k2.lower()
        if not k2:
            continue
        if len(k2) < min_len:
            continue
        if max_len is not None and len(k2) > max_len:
            continue
        key = k2.casefold()
        if dedupe and key in seen:
            continue
        seen.add(key)
        out.append(k2)
    return out

def _join_keywords(kws: List[str]) -> str:
    # For ROUGE we compare concatenated strings
    return "; ".join([_clean_kw(k) for k in kws if _clean_kw(k)])

def rouge_scores(
    preds: List[List[str]],
    refs: List[List[str]],
    *,
    use_stemmer: bool = True
) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=use_stemmer)
    r1, rL = [], []
    for p, r in zip(preds, refs):
        pred_text = _join_keywords(p).lower()
        ref_text  = _join_keywords(r).lower()
        s = scorer.score(ref_text, pred_text)
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    n = max(1, len(r1))
    return {
        "rouge1": sum(r1) / n,
        "rougeL": sum(rL) / n,
        "rouge_avg": (sum(r1) + sum(rL)) / (2 * n),
        "count": n,
    }
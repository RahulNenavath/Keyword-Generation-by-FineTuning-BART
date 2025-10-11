import re
import pandas as pd
from typing import List, Dict, Union
from datasets import Dataset, DatasetDict, Features, Value, Sequence

def normalize_kw_string(s: str) -> List[str]:
    """
    Normalize a model output string into a set/list of keywords.
    Splits on newline/semicolon/commas, trims, lowercases, dedups (order-insensitive for metrics).
    """
    if s is None:
        return []
    # split on common separators
    raw = []
    for chunk in s.split("\n"):
        raw.extend([p for w in chunk.split(";") for p in w.split(",")])
    norm = []
    seen = set()
    for k in raw:
        k2 = k.strip()
        k_norm = k2.casefold()
        if k2 and k_norm not in seen:
            seen.add(k_norm)
            norm.append(k2)
    return norm


def f1_keywords(preds: List[List[str]], refs: List[List[str]]) -> Dict[str, float]:
    """
    Set-based precision/recall/F1 across a corpus.
    """
    tp = fp = fn = 0
    for p, r in zip(preds, refs):
        ps = set([x.casefold() for x in p])
        rs = set([x.casefold() for x in r])
        tp += len(ps & rs)
        fp += len(ps - rs)
        fn += len(rs - ps)
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

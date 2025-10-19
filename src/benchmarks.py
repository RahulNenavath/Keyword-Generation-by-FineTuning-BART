import argparse
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from utils import rouge_scores
from typing import List, Optional
from schema import YakeConfig, KeyBertConfig
from models import YakeExtractor, KeyBertExtractor

def run_benchmark(
    ds: Dataset,
    yake_cfg: YakeConfig,
    kb_cfg: KeyBertConfig,
) -> None:
    texts = ds["text"]
    refs  = ds["keywords"]

    # YAKE
    ye = YakeExtractor(yake_cfg)
    preds_yake: List[List[str]] = []
    for t in tqdm(texts, desc="YAKE"):
        preds_yake.append(ye.extract(t))

    # KeyBERT
    ke = KeyBertExtractor(kb_cfg)
    preds_kb: List[List[str]] = []
    for t in tqdm(texts, desc="KeyBERT"):
        preds_kb.append(ke.extract(t))

    # ROUGE
    m_yake = rouge_scores(preds_yake, refs)
    m_kb   = rouge_scores(preds_kb, refs)

    # Report
    print("\n=== Benchmark (ROUGE F1) ===")
    print(f"Samples: {m_yake['count']}")
    print("\nYAKE:")
    print(f"  ROUGE-1: {m_yake['rouge1']:.4f}")
    print(f"  ROUGE-L: {m_yake['rougeL']:.4f}")
    print(f"  Avg    : {m_yake['rouge_avg']:.4f}")

    print("\nKeyBERT:")
    print(f"  ROUGE-1: {m_kb['rouge1']:.4f}")
    print(f"  ROUGE-L: {m_kb['rougeL']:.4f}")
    print(f"  Avg    : {m_kb['rouge_avg']:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark YAKE and KeyBERT keyword extraction.")
    
    # YAKE params
    p.add_argument("--yake-top-k", type=int, default=20)
    p.add_argument("--yake-max-ngram", type=int, default=3)
    p.add_argument("--yake-dedup", type=float, default=0.9)

    # KeyBERT params
    p.add_argument("--kb-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--kb-top-k", type=int, default=20)
    p.add_argument("--kb-ngram-min", type=int, default=1)
    p.add_argument("--kb-ngram-max", type=int, default=3)
    p.add_argument("--kb-mmr", action="store_true", default=True)
    p.add_argument("--kb-diversity", type=float, default=0.6)

    return p.parse_args()

def main():
    args = parse_args()
    dsd = load_from_disk("Data/keyword_dataset")
    test_set_ds = dsd['test']

    yake_cfg = YakeConfig(
        top_k=args.yake_top_k,
        max_ngram_size=args.yake_max_ngram,
        dedup_thresh=args.yake_dedup,
    )
    
    kb_cfg = KeyBertConfig(
        model_name=args.kb_model,
        top_k=args.kb_top_k,
        keyphrase_ngram_range=(args.kb_ngram_min, args.kb_ngram_max),
        use_mmr=args.kb_mmr,
        diversity=args.kb_diversity,
    )
    run_benchmark(test_set_ds, yake_cfg, kb_cfg)

if __name__ == "__main__":
    main()
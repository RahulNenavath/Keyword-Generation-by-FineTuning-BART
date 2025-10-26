import argparse
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from utils import rouge_scores
from typing import List, Optional
from schema import YakeConfig, KeyBertConfig, StageTwoLLMConfig
from models import YakeExtractor, KeyBertExtractor
from two_stage import TwoStageKeywordGenerator

def run_benchmark(
    ds: Dataset,
    yake_cfg: YakeConfig,
    kb_cfg: KeyBertConfig,
    two_stage_cfg: Optional[StageTwoLLMConfig] = None,
    max_samples: Optional[int] = None,
) -> None:
    texts = ds["text"]
    refs  = ds["keywords"]
    if max_samples is not None:
        texts = texts[:max_samples]
        refs = refs[:max_samples]

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
    
    # Two-Stage LLM
    tsg = TwoStageKeywordGenerator(stage_two_cfg=two_stage_cfg)
    preds_two: List[List[str]] = []
    for t in tqdm(texts, desc="Two-Stage LLM"):
        preds_two.append(tsg.generate(t))

    # ROUGE
    m_yake = rouge_scores(preds_yake, refs)
    m_kb   = rouge_scores(preds_kb, refs)
    m_two = rouge_scores(preds_two, refs)

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

    print("\nTwo-Stage LLM:")
    print(f"  ROUGE-1: {m_two['rouge1']:.4f}")
    print(f"  ROUGE-L: {m_two['rougeL']:.4f}")
    print(f"  Avg    : {m_two['rouge_avg']:.4f}")
        

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark YAKE, KeyBERT, and optional Two-Stage LLM refinement.")
    
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

    # Two-Stage options
    p.add_argument("--two-stage", action="store_true", help="Run two-stage candidate+LLM refinement (requires mlx-lm).")
    p.add_argument("--llm-model-id", type=str, default="mlx-community/Qwen3-0.6B-8bit")
    p.add_argument("--llm-max-new", type=int, default=256)
    p.add_argument("--llm-temp", type=float, default=0.0)
    p.add_argument("--llm-top-p", type=float, default=0.9)
    p.add_argument("--llm-rep-penalty", type=float, default=1.05)
    p.add_argument("--max-samples", type=int, default=50, help="Limit the number of test samples for quick runs.")

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
    
    two_stage_cfg = None
    if args.two_stage:
        two_stage_cfg = StageTwoLLMConfig(
            model_id=args.llm_model_id,
            max_new_tokens=args.llm_max_new,
            temperature=args.llm_temp,
            top_p=args.llm_top_p,
            repetition_penalty=args.llm_rep_penalty,
        )

    run_benchmark(test_set_ds, yake_cfg, kb_cfg, two_stage_cfg=two_stage_cfg, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
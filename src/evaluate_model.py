import argparse
from typing import List

from datasets import load_from_disk, Dataset

from schema import Seq2SeqKwInferConfig
from models import Seq2SeqKeywordGenerator
from utils import rouge_scores, f1_keywords


def batched(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def evaluate_model(model_dir: str, dataset_path: str, max_samples: int | None, batch_size: int) -> None:
    dsd = load_from_disk(dataset_path)
    ds: Dataset = dsd["test"]

    texts = ds["text"]
    refs = ds["keywords"]
    if max_samples is not None:
        texts = texts[:max_samples]
        refs = refs[:max_samples]

    cfg = Seq2SeqKwInferConfig(
        model_dir=model_dir,
        prefix="Extract keywords:\n\n",
        sep="; ",
        batch_size=batch_size,
    )
    gen = Seq2SeqKeywordGenerator(cfg)

    preds: List[List[str]] = []
    for chunk in batched(texts, batch_size):
        preds.extend(gen.generate(chunk, return_as_lists=True))

    r = rouge_scores(preds, refs)
    f = f1_keywords(preds, refs)

    print("\n=== Seq2Seq Model Evaluation ===")
    print(f"Samples: {r['count']}")
    print("ROUGE-1 F1:", f"{r['rouge1']:.4f}")
    print("ROUGE-L F1:", f"{r['rougeL']:.4f}")
    print("ROUGE Avg:", f"{r['rouge_avg']:.4f}")
    print("F1 (set overlap):", f"{f['f1']:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned seq2seq keyword generator on the test split.")
    p.add_argument("--model-dir", required=True, help="Path to directory containing the saved model & tokenizer.")
    p.add_argument(
        "--dataset-path",
        default="Data/keyword_dataset",
        help="Path to load_from_disk dataset with 'train/validation/test' splits.",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples for quick evaluation.")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

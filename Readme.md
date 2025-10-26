# Keyword Generation by Fine‑Tuning Seq2Seq (FLAN‑T5)

This project fine‑tunes a seq2seq transformer (default: FLAN‑T5) to generate keyword phrases from text. It also includes strong keyword extraction baselines (YAKE, KeyBERT) and utilities to prepare HF datasets and evaluate with ROUGE.

Highlights
- End‑to‑end training with Hugging Face Trainer (seq2seq)
- Clean preprocessing and dynamic padding with fast tokenizers
- Baselines: YAKE and KeyBERT with a simple benchmarking script
- Evaluation script to compute ROUGE‑1/ROUGE‑L and set‑overlap F1 on test split

## Setup

Conda (recommended)
1. conda env create -f environment.yml
2. conda activate keywordgen

Alternatively, pip users can install from `requirements.txt` (ensure PyTorch is installed with the right build for your hardware).

## Data

Place a Hugging Face dataset (load_from_disk format) at `Data/keyword_dataset` with splits `train/validation/test` and fields:
- text: string
- keywords: List[string]

The provided `Data/keyword_dataset/` folder is an example artifact. You can also build from a TSV/Parquet using the notebooks or your own script.

## Train

Use the seq2seq training pipeline in `src/train.py` (class `KeywordSeq2SeqModel`). A minimal example is in `src/train.py` under the `main_train.py` section.

Steps
- Ensure your dataset path is correct (default: `Data/keyword_dataset`).
- Adjust `ModelConfig` (e.g., `google/flan-t5-base`) and `TrainConfig` as needed.
- Run the script to start training; checkpoints will be saved under the configured `output_dir`.

Artifacts
- `runs/<name>/checkpoint-*` — intermediate checkpoints
- `runs/<name>/` — final model and tokenizer (if saved at end)

## Inference

Use `src/inference.py` with `Seq2SeqKwInferConfig` to load a saved model directory and generate keywords.
Key notes
- The `prefix` must match what was used during training (default: `"Extract keywords:\n\n"`).
- `sep` should match how targets were joined during training (default: `"; "`).

## Evaluation (ROUGE)

We provide `src/evaluate_model.py` to compute ROUGE‑1/ROUGE‑L (F1) and a set‑overlap F1 for quick sanity checks.

Example
- python src/evaluate_model.py --model-dir runs/flan_t5_kw_cuda/checkpoint-500 --dataset-path Data/keyword_dataset --max-samples 200 --batch-size 16

This prints ROUGE‑1, ROUGE‑L, average, and set‑overlap F1.

## Baselines: YAKE & KeyBERT

Use `src/benchmarks.py` to run baseline extractors against the test split and report ROUGE.
Notes
- KeyBERT downloads a SentenceTransformer model (default `all-MiniLM-L6-v2`).
- YAKE has configurable n‑grams, deduplication, and top‑K.

## Notes

- The previous README referenced BART; the current codebase defaults to FLAN‑T5. You can switch to BART by updating `model_id` in `schema.ModelConfig`, but ensure tokenizer and padding settings are appropriate for that model.
- If you plan to continue LoRA/PEFT training from checkpoints, refer to the notebook utilities or integrate an adapter‑aware loader; the seq2seq path here focuses on standard fine‑tuning.


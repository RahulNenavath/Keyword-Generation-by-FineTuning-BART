# Keyword Generation: Extraction and Generation Pipelines

This repository tackles automatic keyword generation for documents using both unsupervised extractors and supervised seq2seq generation. It also includes a two‑stage pipeline that refines extractor candidates with a lightweight local LLM.

## Problem statement

Given an input document, produce a concise, non‑redundant list of keywords/keyphrases that capture the major topics and important concepts of the document.

## Expected input and output

- Input: a document (string)
- Output: a list of strings (keywords/keyphrases)

Dataset format (Hugging Face load_from_disk):
- `text`: string
- `keywords`: List[string]

We expect the dataset at `Data/keyword_dataset` with splits `train`, `validation`, and `test`.

## Approaches included

1) YAKE (unsupervised extractor)
- Implementation: `YakeExtractor` in `src/models.py`
- Tunable: top‑K, n‑gram size, dedup threshold, etc. (`YakeConfig`)

2) KeyBERT (unsupervised extractor)
- Implementation: `KeyBertExtractor` in `src/models.py`
- Uses SentenceTransformers to embed candidate phrases; supports MMR for diversity (`KeyBertConfig`)

3) Fine‑tuned FLAN‑T5 (seq2seq generation)
- Training code: `src/train.py` (`KeywordSeq2SeqModel`)
- Preprocessing and collator: `src/dataloaders.py`
- Inference wrapper: `src/models.py` (`Seq2SeqKeywordGenerator`)
- Evaluation helper: `src/evaluate_model.py`

4) Two‑Stage pipeline (candidates + LLM refinement)
- Stage One: run YAKE, KeyBERT (and optionally TextRank) to create candidate keywords
- Stage Two: refine the candidate list using a local LLM via MLX (`src/two_stage.py`)
- Prompts are configurable in `src/prompt-template.toml` and loaded through `src/config.py` → `schema.StageTwoLLMConfig`
- Notes:
	- Requires Apple Silicon and `mlx-lm` for local LLM inference
	- You can set the LLM checkpoint via CLI (e.g., “Llama 3.2 3B”); defaults can be changed in `StageTwoLLMConfig`
	- TextRank is optional; if you add spaCy + pytextrank, replace the stub `TextRankExtractor` with a real implementation

## Setup

Conda (recommended)
1. conda env create -f environment.yml
2. conda activate keywordgen

Alternatively, pip users can install from `requirements.txt` (ensure PyTorch and CUDA/MPS builds match your hardware). Two‑stage LLM refinement requires `mlx-lm` (Apple Silicon).

## Train (FLAN‑T5)

Use `src/train.py` and the `KeywordSeq2SeqModel` class. A runnable example is provided in the file under the `main_train.py` section.

Artifacts
- `runs/<name>/checkpoint-*` — intermediate checkpoints
- `runs/<name>/` — final model and tokenizer (if saved at end)

## Inference (FLAN‑T5)

Use `src/inference.py` with `Seq2SeqKwInferConfig` to load a saved model directory and generate keywords.
Key notes
- `prefix` must match training (default: `"Extract keywords:\n\n"`)
- `sep` should match how targets were joined (default: `"; "`)

## Evaluation (ROUGE)

Use `src/evaluate_model.py` to compute ROUGE‑1, ROUGE‑L (F1), and a set‑overlap F1 on the test split.

Example
```bash
python src/evaluate_model.py --model-dir runs/flan_t5_kw_cuda/checkpoint-500 --dataset-path Data/keyword_dataset --max-samples 200 --batch-size 16
```

## Benchmarking

Run `src/benchmarks.py` to evaluate extractors and optionally the two‑stage pipeline on the test split.

Examples
```bash
# YAKE + KeyBERT only
python src/benchmarks.py --max-samples 500

# Add Two‑Stage refinement (requires mlx-lm; Apple Silicon)
python src/benchmarks.py --two-stage --llm-model-id meta-llama/Llama-3.2-3B-Instruct --max-samples 200
```

### Results (ROUGE F1)

Using the project’s test split, we observed the following scores:

YAKE:
- ROUGE-1: 0.2429
- ROUGE-L: 0.1940
- Avg: 0.2185

KeyBERT:
- ROUGE-1: 0.2524
- ROUGE-L: 0.1553
- Avg: 0.2038

Flan-T5 encoder-decoder:
- ROUGE-1: 0.3854
- ROUGE-L: 0.3207
- Avg: 0.3531

Two-stage Keyword Generation with Llama 3.2 3B model:
- ROUGE-1: 0.3172
- ROUGE-L: 0.2334
- Avg: 0.2753

Notes
- Two‑stage performance depends heavily on the base candidate quality and prompt/LLM choice. The above numbers used Llama 3.2 3B for refinement.
- FLAN‑T5 fine‑tuning provides the strongest results on this dataset among the included methods.

## Repo map (selected)

- `src/dataloaders.py` — Preprocessing and simple data collator for seq2seq
- `src/train.py` — Training pipeline for FLAN‑T5
- `src/models.py` — Inference wrapper for seq2seq; YAKE/KeyBERT (and TextRank stub)
- `src/two_stage.py` — Two‑stage candidate generation + LLM refinement
- `src/schema.py` — Config dataclasses
- `src/utils.py` — Normalization, ROUGE computation, dataset helpers
- `src/evaluate_model.py` — Evaluate a saved seq2seq model on test split
- `src/benchmarks.py` — Benchmark extractors and optional two‑stage pipeline
- `src/config.py` — Loads .env and TOML prompts (`prompt-template.toml`)



from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class Seq2SeqKwPreprocConfig:
    prefix: str = "Extract keywords:\n\n"   # short instruction shown to encoder
    sep: str = "; "                         # how to join keywords into a target string
    max_source_len: int = 1024              # encoder max length (document side)
    max_target_len: int = 128               # decoder max length (keywords side)
    add_eos: bool = True                    # for T5, EOS is usually id=1
    truncate_long_docs: bool = True         # hard-truncate inputs if longer than max_source_len

@dataclass
class ModelConfig:
    model_id: str = "google/flan-t5-base"
    # If your model lacks decoder_start_token_id, use pad_token_id
    force_decoder_start_as_pad: bool = True # ← Enable this for T5 models and disable for non-T5

@dataclass
class TrainConfig:
    output_dir: str = "runs/flan_t5_kw"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 10
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2

    fp16: bool = False
    bf16: bool = False

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    predict_with_generate: bool = True
    generation_max_length: int = 128
    generation_num_beams: int = 4

    report_to: Optional[str] = "none"
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
 
@dataclass
class Seq2SeqKwInferConfig:
    """
    Inference configuration for seq2seq keyword generation.
    """
    # Directory containing the fine-tuned model + tokenizer (HF save_pretrained format)
    model_dir: str

    # Must match your training preprocessor prompt exactly
    prefix: str = "Extract keywords:\n\n"   # e.g., "Extract keywords:\n\n"
    sep: str = "; "                         # how keywords were joined during training

    # Generation settings
    max_new_tokens: int = 64
    num_beams: int = 4
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    # Runtime settings
    device_preference: Optional[str] = None      # "cuda" | "mps" | "cpu" | None (auto)
    torch_dtype: Optional[str] = None            # "float16" | "bfloat16" | "float32" | None (auto)
    batch_size: int = 8
    truncate_source_to: Optional[int] = 2048     # cap encoder length for very long docs (None = no cap)


@dataclass
class YakeConfig:
    language: str = "en"            # ISO code, e.g., 'en', 'de'
    max_ngram_size: int = 3         # 1..3 typical
    dedup_thresh: float = 0.9       # YAKE’s internal surface dedup threshold
    window_size: int = 1            # context window
    top_k: int = 20                 # how many candidates to return
    features: Optional[List[str]] = None  # default YAKE feature set
    # post-processing
    lowercase: bool = False
    min_len: int = 2
    max_len: Optional[int] = None
    dedupe: bool = True


@dataclass
class KeyBertConfig:
    model_name: str = "all-MiniLM-L6-v2"     # SentenceTransformer backbone
    top_k: int = 20
    keyphrase_ngram_range: Tuple[int, int] = (1, 3)
    stop_words: Optional[str] = "english"    # or None
    use_maxsum: bool = False                 # diversification strategy A
    use_mmr: bool = True                     # diversification strategy B (MMR)
    diversity: float = 0.6                   # 0..1 (only if MMR enabled)
    min_df: float | int = 1                  # (optional) doc frequency filter for candidates
    # post-processing
    lowercase: bool = False
    min_len: int = 2
    max_len: Optional[int] = None
    dedupe: bool = True
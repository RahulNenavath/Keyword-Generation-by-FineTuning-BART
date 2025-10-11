from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    # Provide an already-prepared DatasetDict OR instruct how to load/format one.
    hf_path_or_none: Optional[str] = None  # e.g. path from save_to_disk OR HF hub id
    text_field: str = "text"
    keywords_field: str = "keywords"  # list[str]
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 1000  # to keep eval fast


@dataclass
class PromptConfig:
    # Generic prompt template; keep KEYWORDS: as the "response tag" for the collator
    system_preamble: str = (
        "You are an expert keyword generator. "
        "Extract concise, relevant keywords for the document below."
    )
    response_tag: str = "KEYWORDS:"
    sep: str = "; "  # how labels are joined for the target text


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "runs/kwgen"
    load_in_4bit: bool = False          # QLoRA path
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # if None, auto-detect common names
    max_seq_len: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    seed: int = 42
    bf16: bool = False
    fp16: bool = True
    report_to: Optional[str] = None  # "wandb" or None
    gradient_checkpointing: bool = True
    save_total_limit: int = 2
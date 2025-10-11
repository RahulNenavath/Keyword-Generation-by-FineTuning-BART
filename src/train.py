import os
import math
import json
import torch
import random
from config import Config
from typing import Dict, Optional
from utils import normalize_kw_string, f1_keywords
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments
from schema import TrainConfig, PromptConfig, DataConfig
from dataloaders import KeywordDataModule, KeywordCompletionCollator
from models import QLoRALoader
from trl import SFTTrainer, SFTConfig


class KwGenTrainer:
    def __init__(self, train_cfg: TrainConfig, prompt_cfg: PromptConfig):
        self.cfg = train_cfg
        self.prompt = prompt_cfg
        self.trainer = None

    def build(self, tok, model, dsd_mapped: DatasetDict):
        collator = KeywordCompletionCollator(
            tokenizer=tok,
            response_template=self.prompt.response_tag,  # "KEYWORDS:"
            max_length=self.cfg.max_seq_len,
            pad_to_multiple_of=8,
        )

        sft_args = SFTConfig(
            output_dir=self.cfg.output_dir,
            packing=False,                          
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_train_epochs=self.cfg.num_train_epochs,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type="cosine",
            logging_steps=self.cfg.logging_steps,
            eval_strategy="steps",
            eval_steps=self.cfg.eval_steps,
            save_steps=self.cfg.save_steps,
            warmup_ratio=self.cfg.warmup_ratio,
            weight_decay=self.cfg.weight_decay,
            bf16=self.cfg.bf16,
            fp16=self.cfg.fp16,
            report_to=self.cfg.report_to,
            seed=self.cfg.seed,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            save_total_limit=self.cfg.save_total_limit,
        )

        self.trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dsd_mapped.get("train"),
            eval_dataset=dsd_mapped.get("validation"),
            data_collator=collator,
        )
        return self.trainer

    @torch.no_grad()
    def evaluate_keywords(self, tok, model, eval_ds: Dataset, sample_size: int = 512, gen_kwargs=None) -> Dict[str, float]:
        """
        Generates keywords for a subset of the eval set and computes F1/Precision/Recall.
        """
        if gen_kwargs is None:
            gen_kwargs = dict(
                max_new_tokens=96, temperature=0.2, do_sample=False, top_p=1.0, repetition_penalty=1.05
            )

        # Take a subset to keep eval fast
        n = min(sample_size, len(eval_ds))
        idxs = list(range(n))
        prompts = [eval_ds[i]["text"] for i in idxs]
        refs = [eval_ds[i]["labels"] for i in idxs]  # joined with sep in DataModule

        device = model.device
        preds_kw, refs_kw = [], []
        model.eval()

        for p, r in zip(prompts, refs):
            inputs = tok(p, return_tensors="pt").to(device)
            out = model.generate(**inputs, **gen_kwargs)
            gen_txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            preds_kw.append(normalize_kw_string(gen_txt))
            refs_kw.append(normalize_kw_string(r))

        metrics = f1_keywords(preds_kw, refs_kw)
        return metrics

    def train_and_eval(self, tok, model, dsd_mapped: DatasetDict) -> Dict[str, float]:
        self.build(tok, model, dsd_mapped)
        self.trainer.train()
        # quick eval
        if "validation" in dsd_mapped and len(dsd_mapped["validation"]) > 0:
            metrics = self.evaluate_keywords(tok, self.trainer.model, dsd_mapped["validation"])
        else:
            metrics = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
        # Save adapter & tokenizer
        adapter_dir = os.path.join(self.cfg.output_dir, "adapter")
        self.trainer.model.save_pretrained(adapter_dir)
        # The tokenizer object was created/returned by the loader as `tok`.
        # SFTTrainer may not expose `tokenizer` as an attribute, so save
        # the tokenizer passed into this method instead.
        tok.save_pretrained(adapter_dir)

        with open(os.path.join(self.cfg.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    
class KeywordGenPipeline:
    def __init__(self, data_cfg: DataConfig, prompt_cfg: PromptConfig, train_cfg: TrainConfig):
        self.data_cfg = data_cfg
        self.prompt_cfg = prompt_cfg
        self.train_cfg = train_cfg

        self.data_module = KeywordDataModule(data_cfg, prompt_cfg)
        self.loader = QLoRALoader(train_cfg)
        self.runner = KwGenTrainer(train_cfg, prompt_cfg)

    def run(self, dsd: Optional[DatasetDict] = None) -> Dict[str, float]:
        # Load + format
        dsd_mapped = self.data_module.load(dsd)
        # Load model/tokenizer with QLoRA
        tok, model = self.loader.load()
        # Train + eval + save
        metrics = self.runner.train_and_eval(tok, model, dsd_mapped)
        print("Final metrics:", metrics)
        return metrics


if __name__ == "__main__":
    configobj = Config()
    data_cfg = DataConfig(
        hf_path_or_none=configobj.data_dir / 'keyword_dataset',
        text_field="text",
        keywords_field="keywords",
        max_train_samples=None,
        max_eval_samples=800,
    )
    prompt_cfg = PromptConfig(
        system_preamble="You are an expert keyword generator. Extract concise, relevant keywords for the document below.",
        response_tag="KEYWORDS:",
        sep="; ",
    )
    train_cfg = TrainConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",   # swap here to try other models (Phi-3.5, Llama-3.2-3B, Gemma-2-2B, etc.)
        output_dir="runs/qwen25_3b_kwgen",
        load_in_4bit=True,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=None,                   # auto-detects q_proj/k_proj/... if None
        max_seq_len=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        warmup_ratio=0.03,
        weight_decay=0.0,
        seed=42,
        bf16=False, fp16=True,
        report_to=None,
        gradient_checkpointing=True,
        save_total_limit=2,
    )

    pipe = KeywordGenPipeline(data_cfg, prompt_cfg, train_cfg)
    pipe.run()
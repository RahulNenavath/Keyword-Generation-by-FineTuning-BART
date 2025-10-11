import torch
from schema import TrainConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import List
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class QLoRALoader:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.response_template = cfg  # placeholder to keep interface consistent

    def _guess_target_modules(self, model: torch.nn.Module) -> List[str]:
        # Common linear projection names across LLaMA/Qwen/Gemma/Phi families
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = []
        for n, _ in model.named_modules():
            for c in candidates:
                if n.endswith(c):
                    present.append(c)
        # de-duplicate but keep order
        seen, uniq = set(), []
        for x in present:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq or ["q_proj", "k_proj", "v_proj", "o_proj"]

    def load(self):
        print(f"Loading tokenizer: {self.cfg.model_id}")
        tok = AutoTokenizer.from_pretrained(self.cfg.model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        print(f"Loading model: {self.cfg.model_id}")
        dtype = torch.bfloat16 if self.cfg.bf16 else torch.float16
    

        # quant_cfg = None
        # if self.cfg.load_in_4bit:
        #     quant_cfg = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.bfloat16 if self.cfg.bf16 else torch.float16,
        #     )
        # dtype = torch.bfloat16 if self.cfg.bf16 else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            dtype=dtype,
            device_map="auto",
        ).to("cuda")

        # if self.cfg.load_in_4bit:
        #     model = prepare_model_for_kbit_training(model)

        target_modules = self.cfg.target_modules or self._guess_target_modules(model)
        print("Using LoRA target_modules:", target_modules)
        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        self.tokenizer = tok
        self.model = model
        return tok, model
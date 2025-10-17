# main_train.py (example usage)
import torch
from datasets import load_from_disk

from schema import Seq2SeqKwPreprocConfig, ModelConfig, TrainConfig
from models import KeywordSeq2SeqModel

if __name__ == "__main__":
    # 1) Load your HF dataset (already prepared by prepare-hf-data.py)
    dsd = load_from_disk("Data/keyword_dataset")  # adjust to your actual path

    # 2) Build configs
    preproc_cfg = Seq2SeqKwPreprocConfig(
        prefix="Extract keywords:\n\n",
        sep="; ",
        max_source_len=1024,
        max_target_len=128,
        add_eos=True,
        truncate_long_docs=True,
    )

    model_cfg = ModelConfig(
        model_id="google/flan-t5-base",
        force_decoder_start_as_pad=True,
    )

    train_cfg = TrainConfig(
        output_dir="runs/flan_t5_kw_cuda",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=15,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        fp16=False,  # keep False unless you're on CUDA with fp16 support
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        report_to="none",
        seed=42,
        early_stopping_patience=5,
        early_stopping_threshold=0.0,
    )

    print(f"CUDA available: {torch.cuda.is_available()}")
    # (Don’t manually .to(device); the Trainer will handle devices.)

    # 3) Build, prepare (tokenize), and train
    trainer = KeywordSeq2SeqModel(dsd, preproc_cfg, model_cfg, train_cfg)
    trainer.prepare(num_proc=4)  # set to None on Windows
    out = trainer.train()
    print("Training summary:", out)
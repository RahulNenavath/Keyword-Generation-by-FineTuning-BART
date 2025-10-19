# train.py
import os
import torch
import yake
from keybert import KeyBERT
from typing import List, Union, Optional, Dict, Any, Tuple
from dataclasses import asdict
from utils import _postprocess
from sentence_transformers import SentenceTransformer

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    GenerationConfig
)

from utils import normalize_kw_string
from schema import (
    Seq2SeqKwPreprocConfig, 
    ModelConfig, 
    TrainConfig, 
    Seq2SeqKwInferConfig,
    YakeConfig,
    KeyBertConfig
    )
from dataloaders import Seq2SeqKeywordPreprocessor, DataCollatorForSeq2SeqSimple


class KeywordSeq2SeqModel:
    """
    End-to-end trainer for keyword generation (seq2seq).
    - Consumes a Hugging Face DatasetDict with splits: 'train', 'validation', 'test'
    - Uses your preprocessor & collator from dataloader.py
    - Builds a Seq2SeqTrainer with early stopping
    """

    def __init__(
        self,
        dsd: DatasetDict,
        preproc_cfg: Seq2SeqKwPreprocConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
    ):
        if not isinstance(dsd, DatasetDict):
            raise ValueError("`dsd` must be a Hugging Face DatasetDict with train/validation/test.")

        self.dsd_raw = dsd
        self.preproc_cfg = preproc_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.tokenizer = None
        self.model = None
        self.dsd_tok: Optional[DatasetDict] = None
        self.trainer: Optional[Seq2SeqTrainer] = None

    # ---------- build components ----------
    def _build_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_id, use_fast=True)

        # T5/FLAN-T5 specifics
        # Ensure pad token exists & pad on the right
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            # FLAN-T5 uses pad_token_id = 0; fallback to eos if needed
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_cfg.model_id)

        # Make sure decoder_start_token_id is properly set for encoder-decoder generation
        if self.model_cfg.force_decoder_start_as_pad:
            if getattr(self.model.config, "decoder_start_token_id", None) is None:
                self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id

        # Align config with tokenizer (prevents those BOS/EOS/PAD warnings)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # T5 typically doesn't use bos_token; safe to leave unset

    def _preprocess_dataset(self, num_proc: Optional[int] = None):
        pp = Seq2SeqKeywordPreprocessor(self.tokenizer, self.preproc_cfg)
        self.dsd_tok = pp.apply(self.dsd_raw, num_proc=num_proc)

    def _build_trainer(self) -> Seq2SeqTrainer:
        # Collator (dynamic padding + label masking)
        collator = DataCollatorForSeq2SeqSimple(tokenizer=self.tokenizer, pad_to_multiple_of=None)

        # Training args
        args = Seq2SeqTrainingArguments(
            output_dir=self.train_cfg.output_dir,
            per_device_train_batch_size=self.train_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            num_train_epochs=self.train_cfg.num_train_epochs,
            learning_rate=self.train_cfg.learning_rate,
            lr_scheduler_type=self.train_cfg.lr_scheduler_type,
            warmup_ratio=self.train_cfg.warmup_ratio,
            weight_decay=self.train_cfg.weight_decay,

            logging_steps=self.train_cfg.logging_steps,
            eval_strategy=self.train_cfg.eval_strategy,
            eval_steps=self.train_cfg.eval_steps,
            save_steps=self.train_cfg.save_steps,
            save_total_limit=self.train_cfg.save_total_limit,

            fp16=self.train_cfg.fp16,
            bf16=self.train_cfg.bf16,

            load_best_model_at_end=self.train_cfg.load_best_model_at_end,
            metric_for_best_model=self.train_cfg.metric_for_best_model,
            greater_is_better=self.train_cfg.greater_is_better,

            predict_with_generate=self.train_cfg.predict_with_generate,
            generation_max_length=self.train_cfg.generation_max_length,
            generation_num_beams=self.train_cfg.generation_num_beams,

            report_to=self.train_cfg.report_to,
            seed=self.train_cfg.seed,
        )

        # NOTE: We pass `processing_class=self.tokenizer` to avoid the tokenizer deprecation warning.
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dsd_tok.get("train"),
            eval_dataset=self.dsd_tok.get("validation"),
            data_collator=collator,
            processing_class=self.tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_cfg.early_stopping_patience,
                    early_stopping_threshold=self.train_cfg.early_stopping_threshold,
                )
            ],
        )
        return trainer

    def prepare(self, num_proc: Optional[int] = None):
        """Build tokenizer+model and preprocess datasets (tokenize)."""
        self._build_tokenizer_and_model()
        self._preprocess_dataset(num_proc=num_proc)
        self.trainer = self._build_trainer()

    def train(self) -> Dict[str, Any]:
        """
        Launch training; returns final evaluation metrics on the validation set.
        Saves:
          - best model (if load_best_model_at_end=True in TrainConfig)
          - tokenizer
          - trainer state
        """
        if self.trainer is None:
            raise RuntimeError("Call `.prepare()` before `.train()`.")

        # Let Trainer handle device placement (CUDA/MPS/CPU)
        train_output = self.trainer.train()

        # Save final artifacts
        self.trainer.save_model(self.train_cfg.output_dir)        # saves model weights
        self.tokenizer.save_pretrained(self.train_cfg.output_dir) # saves tokenizer

        # Evaluate on validation split at end (optional)
        metrics = self.trainer.evaluate(self.dsd_tok.get("validation"))
        # Persist metrics to disk, too
        metrics_path = os.path.join(self.train_cfg.output_dir, "val_metrics.json")
        try:
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass

        # Return both train summary & val metrics for convenience
        return {
            "train": {
                "global_step": getattr(train_output, "global_step", None),
                "training_loss": getattr(train_output, "training_loss", None),
            },
            "validation": metrics,
            "configs": {
                "model": asdict(self.model_cfg),
                "train": asdict(self.train_cfg),
                "preproc": asdict(self.preproc_cfg),
            },
        }


class Seq2SeqKeywordGenerator:
    """
    Inference wrapper for seq2seq keyword generation (e.g., FLAN-T5).
    Loads a fine-tuned model from `cfg.model_dir` and generates keyword lists.
    """

    def __init__(self, cfg: Seq2SeqKwInferConfig):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device_preference)
        self.dtype = self._pick_dtype(cfg.torch_dtype)

        # Load tokenizer first; align PAD/EOS as needed
        self.tok = AutoTokenizer.from_pretrained(cfg.model_dir, use_fast=True)
        self.tok.padding_side = "right"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = getattr(self.tok, "eos_token_id", 0)

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_dir,
            torch_dtype=self.dtype,
            device_map=None,  # we'll place it explicitly below
        )

        # Ensure decoder_start_token_id (T5 uses pad_token_id)
        if getattr(self.model.config, "decoder_start_token_id", None) is None:
            self.model.config.decoder_start_token_id = self.tok.pad_token_id

        # Align config PAD with tokenizer to avoid warnings
        self.model.config.pad_token_id = self.tok.pad_token_id

        self.model.to(self.device)
        self.model.eval()

        self.gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=(cfg.temperature and cfg.temperature > 0.0),
        )

    @torch.inference_mode()
    def generate(
        self,
        documents: Union[str, List[str]],
        return_as_lists: bool = True,
    ) -> Union[List[str], List[List[str]]]:
        """
        Generate keywords for one or many documents.

        Args:
            documents: str or List[str]
            return_as_lists: True => List[List[str]] (normalized keywords)
                             False => List[str] (raw decoded strings)

        Returns:
            For single str input: a single List[str] (or str if return_as_lists=False)
            For list input: List[List[str]] (or List[str])
        """
        is_single = isinstance(documents, str)
        docs = [documents] if is_single else list(documents)

        # Build prompts (must match training format)
        prompts = [self._build_input_text(d) for d in docs]

        # Optional batching if you pass big lists
        results: List[str] = []
        bs = max(1, int(self.cfg.batch_size))

        for start in range(0, len(prompts), bs):
            batch_prompts = prompts[start:start + bs]
            enc = self.tok(
                batch_prompts,
                padding=True,
                truncation=bool(self.cfg.truncate_source_to),
                max_length=self.cfg.truncate_source_to,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outs = self.model.generate(**enc, generation_config=self.gen_cfg)
            decoded = self.tok.batch_decode(outs, skip_special_tokens=True)
            results.extend(decoded)

        if return_as_lists:
            results = [normalize_kw_string(s) for s in results]

        return results[0] if is_single else results

    def _build_input_text(self, doc: str) -> str:
        doc = (doc or "").strip()
        # Matches training: prefix + document
        return f"{self.cfg.prefix}{doc}\n\n"

    def _pick_device(self, pref: Optional[str]) -> torch.device:
        if pref:
            pref = pref.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _pick_dtype(self, name: Optional[str]):
        if name is None:
            # default: fp16 on CUDA, fp32 elsewhere
            return torch.float16 if torch.cuda.is_available() else torch.float32
        name = name.lower()
        if name in ("fp16", "float16", "half"):
            return torch.float16
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32
    
class YakeExtractor:
    """
    Thin wrapper over 'yake' to produce a clean List[str] for a single document.
    """
    def __init__(self, cfg: Optional[YakeConfig] = None):
        
        self.cfg = cfg or YakeConfig()
        self._yake = yake.KeywordExtractor(
            lan=self.cfg.language,
            n=self.cfg.max_ngram_size,
            dedupLim=self.cfg.dedup_thresh,
            windowsSize=self.cfg.window_size,
            top=self.cfg.top_k,
            features=self.cfg.features,
        )

    def extract(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        scored: List[Tuple[str, float]] = self._yake.extract_keywords(text)
        # YAKE returns (keyphrase, score) where lower score is better
        kws_sorted = [kp for kp, _ in sorted(scored, key=lambda x: x[1])]
        return _postprocess(
            kws_sorted,
            lowercase=self.cfg.lowercase,
            min_len=self.cfg.min_len,
            max_len=self.cfg.max_len,
            dedupe=self.cfg.dedupe,
        )

    def extract_many(self, docs: List[str]) -> List[List[str]]:
        return [self.extract(d) for d in docs]
    
class KeyBertExtractor:
    """
    KeyBERT with Sentence-Transformer embeddings + diversification.
    Returns clean List[str].
    """
    def __init__(self, cfg: Optional[KeyBertConfig] = None):
        
        self.cfg = cfg or KeyBertConfig()
        st_model = SentenceTransformer(self.cfg.model_name)
        self._kw = KeyBERT(model=st_model)

    def extract(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        kw_scores = self._kw.extract_keywords(
            text,
            keyphrase_ngram_range=self.cfg.keyphrase_ngram_range,
            stop_words=self.cfg.stop_words,
            top_n=self.cfg.top_k,
            use_maxsum=self.cfg.use_maxsum,
            use_mmr=self.cfg.use_mmr,
            diversity=self.cfg.diversity,
            vectorizer=None,        # let KeyBERT handle candidates, or pass a custom one
            min_df=self.cfg.min_df,
        )
        # KeyBERT returns List[(keyword, score)] where higher score ~ better
        kws_sorted = [kp for kp, _ in kw_scores]
        return _postprocess(
            kws_sorted,
            lowercase=self.cfg.lowercase,
            min_len=self.cfg.min_len,
            max_len=self.cfg.max_len,
            dedupe=self.cfg.dedupe,
        )

    def extract_many(self, docs: List[str]) -> List[List[str]]:
        return [self.extract(d) for d in docs]
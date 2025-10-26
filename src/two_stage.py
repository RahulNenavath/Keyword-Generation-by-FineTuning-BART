from __future__ import annotations

import re
from dataclasses import dataclass
import ast
from typing import List, Dict, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import rouge_scores

from schema import (
    YakeConfig,
    KeyBertConfig,
    TextRankConfig,
    StageTwoLLMConfig,
)
from models import YakeExtractor, KeyBertExtractor, TextRankExtractor


@dataclass
class StageOneResult:
    yake: List[str]
    keybert: List[str]
    textrank: List[str]
    combined: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "yake": self.yake,
            "keybert": self.keybert,
            "textrank": self.textrank,
            "combined": self.combined,
        }


class StageOneCandidateGenerator:
    """
    Runs YAKE, KeyBERT, and TextRank keyword extractors in parallel to gather candidates.
    """

    def __init__(
        self,
        yake_cfg: Optional[YakeConfig] = None,
        keybert_cfg: Optional[KeyBertConfig] = None,
        textrank_cfg: Optional[TextRankConfig] = None,
    ):
        self.yake = YakeExtractor(yake_cfg)
        self.keybert = KeyBertExtractor(keybert_cfg)
        self.textrank = TextRankExtractor(textrank_cfg)

    def extract(self, document: str) -> StageOneResult:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self.yake.extract, document): "yake",
                pool.submit(self.keybert.extract, document): "keybert",
                pool.submit(self.textrank.extract, document): "textrank",
            }
            partial: Dict[str, List[str]] = {"yake": [], "keybert": [], "textrank": []}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    partial[name] = future.result()
                except Exception:
                    partial[name] = []

        combined = self._merge_candidates(
            partial["yake"], partial["keybert"], partial["textrank"]
        )
        return StageOneResult(
            yake=partial["yake"],
            keybert=partial["keybert"],
            textrank=partial["textrank"],
            combined=combined,
        )

    def extract_many(self, documents: List[str]) -> List[StageOneResult]:
        return [self.extract(doc) for doc in documents]

    @staticmethod
    def _merge_candidates(*lists: List[str]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for kw_list in lists:
            for kw in kw_list:
                clean = (kw or "").strip()
                if not clean:
                    continue
                key = clean.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(clean)
        return merged


class StageTwoKeywordRefiner:
    """
    Refines candidate keywords with an MLX-hosted LLM.
    """

    def __init__(self, cfg: Optional[StageTwoLLMConfig] = None):
        self.cfg = cfg or StageTwoLLMConfig()
        self._model = None
        self._tokenizer = None
        self._generate_fn = None

    def refine(self, document: str, candidates: List[str]) -> List[str]:
        if not document:
            return []
        prompt = self._build_prompt(document, candidates)
        response = self._run_model(prompt)
        return response
        # return self._parse_keywords(response)

    def refine_many(self, documents: List[str], candidate_lists: List[List[str]]) -> List[List[str]]:
        return [self.refine(doc, cands) for doc, cands in zip(documents, candidate_lists)]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load, generate  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for StageTwoKeywordRefiner. Install with `pip install mlx-lm`."
            ) from exc

        self._model, self._tokenizer = load(self.cfg.model_id)
        self._generate_fn = generate

    def _build_prompt(self, document: str, candidates: List[str]) -> str:
        self._ensure_model()
        assert self._tokenizer is not None

        candidate_block = self._format_candidates(candidates)
        user_content = self.cfg.user_prompt.format(
            document=document.strip(),
            candidates=candidate_block,
        )
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = (
                f"<|system|>\n{self.cfg.system_prompt}\n</|system|>\n"
                f"<|user|>\n{user_content}\n</|user|>\n<|assistant|>\n"
            )
        return prompt

    def _run_model(self, prompt: str) -> str:
        self._ensure_model()
        assert self._generate_fn is not None
        assert self._model is not None and self._tokenizer is not None

        # Import sampler utilities lazily to avoid hard dependency at module import time
        sampler = None
        logits_processors = None
        try:
            from mlx_lm.sample_utils import make_sampler, make_logits_processors  # type: ignore
            sampler = make_sampler(
                temp=self.cfg.temperature,
                top_p=self.cfg.top_p,
                min_p=0.0,
                min_tokens_to_keep=1,
            )
            logits_processors = make_logits_processors(
                repetition_penalty=self.cfg.repetition_penalty if self.cfg.repetition_penalty else None,
            )
        except Exception:
            sampler = None
            logits_processors = None

        kwargs = {"max_tokens": self.cfg.max_new_tokens}
        if sampler is not None:
            kwargs["sampler"] = sampler
        if logits_processors:
            kwargs["logits_processors"] = logits_processors

        response = self._generate_fn(
            self._model,
            self._tokenizer,
            prompt,
            verbose=False,
            **kwargs,
        )
        return response if isinstance(response, str) else str(response)

    def _parse_keywords(self, text: str) -> List[str]:
        if not text:
            return []

        # Locate the response prefix (case-insensitive, colon optional)
        prefix_pattern = re.compile(
            rf"{re.escape(self.cfg.response_prefix.rstrip(':'))}\s*:?", re.IGNORECASE
        )
        match = prefix_pattern.search(text)
        tail = text[match.end():] if match else text

        lines = [line.strip() for line in tail.splitlines()]
        keywords: List[str] = []
        stop_pattern = re.compile(r"^(?:assistant|analysis|note|explanation)\b", re.IGNORECASE)

        for line in lines:
            if not line:
                if keywords:
                    break
                continue

            cleaned = self._clean_keyword(line)
            if not cleaned:
                continue
            if stop_pattern.match(cleaned):
                break
            # Avoid echoing headers like "Keywords:" that might appear again
            if prefix_pattern.fullmatch(cleaned):
                continue
            if re.search(r"\bkeywords?\b", cleaned, re.IGNORECASE) and len(cleaned.split()) > 3:
                continue
            keywords.append(cleaned)

        deduped: List[str] = []
        seen: set[str] = set()
        for kw in keywords:
            norm = kw.casefold()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(kw)
        return deduped

    def _parse_list_literal(self, text: str) -> List[str]:
        """
        Try to parse a Python list literal of strings from the LLM response.
        The prompt enforces output to be a valid `List[str]`, but we still
        guard for extra tokens by extracting the first [...] block.
        """
        if not text:
            return []
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            snippet = text[start:end + 1]
            obj = ast.literal_eval(snippet)
        except Exception:
            return []
        if not isinstance(obj, list):
            return []
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in obj:
            if not isinstance(item, str):
                continue
            k = self._clean_keyword(item)
            if not k:
                continue
            kn = k.casefold()
            if kn in seen:
                continue
            seen.add(kn)
            cleaned.append(k)
        return cleaned

    @staticmethod
    def _clean_keyword(line: str) -> str:
        line = re.sub(r"^[\-\u2022\*\+\d]+[.)]?\s*", "", line)
        line = line.strip()
        line = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", line)
        line = re.sub(r"[:;\.,]+$", "", line)
        return line.strip()

    @staticmethod
    def _format_candidates(candidates: Iterable[str]) -> str:
        items = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
        if not items:
            return "None supplied."
        return "\n".join(f"- {c}" for c in items)


@dataclass
class TwoStageOutput:
    stage_one: StageOneResult
    refined: List[str]


class TwoStageKeywordGenerator:
    """
    Convenience wrapper that runs both stages sequentially.
    """

    def __init__(
        self,
        stage_one_yake: Optional[YakeConfig] = None,
        stage_one_keybert: Optional[KeyBertConfig] = None,
        stage_one_textrank: Optional[TextRankConfig] = None,
        stage_two_cfg: Optional[StageTwoLLMConfig] = None,
    ):
        self.stage_one = StageOneCandidateGenerator(stage_one_yake, stage_one_keybert, stage_one_textrank)
        self.stage_two = StageTwoKeywordRefiner(stage_two_cfg)

    def generate(self, document: str) -> List[str]:
        stage_one_result = self.stage_one.extract(document)
        response = self.stage_two.refine(document, stage_one_result.combined)
        # Prefer strict list literal parsing; fallback to heuristic parser
        kws = self.stage_two._parse_list_literal(response) or self.stage_two._parse_keywords(response)
        return kws

    def generate_many(self, documents: List[str]) -> List[List[str]]:
        outputs: List[List[str]] = []
        stage_one_results = self.stage_one.extract_many(documents)
        for doc, s1 in zip(documents, stage_one_results):
            response = self.stage_two.refine(doc, s1.combined)
            kws = self.stage_two._parse_list_literal(response) or self.stage_two._parse_keywords(response)
            outputs.append(kws)
        return outputs
    

if __name__ == "__main__":
    # Example usage
    doc = """FlashAttention is a fast, exact attention algorithm that reorders the standard attention computation to be IO-aware. Instead of materializing the large attention matrix in GPU memory, it tiles queries, keys, and values so that blocks fit in on-chip SRAM, drastically reducing reads and writes to high-bandwidth memory. This cut in memory traffic makes attention compute-bound rather than memory-bound on modern accelerators, enabling significant speedups without approximation. The method preserves numerical equivalence to standard softmax attention while supporting long sequences, mixed precision, and dropout. In practice, FlashAttention accelerates training and inference for transformer models in NLP and vision, and it is available through optimized kernels in popular deep learning frameworks. Subsequent work extends the idea with block-sparse layouts, causal masking, and fused kernels for multi-query attention. Although performance depends on sequence length, head dimension, and hardware, users commonly observe 2–4× wall-clock gains with reduced memory footprint, enabling larger context windows and higher batch sizes on the same GPU."""
    expected_keywords = [
        "FlashAttention",
        "IO-aware attention",
        "tiled attention",
        "SRAM tiling",
        "memory bandwidth reduction",
        "compute-bound vs memory-bound",
        "exact softmax attention",
        "long-sequence transformers",
        "GPU optimized kernels",
        "training and inference speedup",
        "reduced memory footprint",
        "causal masking",
        "multi-query attention",
        "block-sparse extensions",
        "mixed precision support",
        "larger context windows",
        "higher batch sizes"
    ]
    two_stage_generator = TwoStageKeywordGenerator()
    predicted_keywords = two_stage_generator.generate(doc)
    
    scores = rouge_scores(preds=[predicted_keywords], refs=[expected_keywords])
    print(scores)

    # print("Stage One Candidates:")
    # print(output.stage_one.to_dict())
    # print("\nRefined Keywords:")
    # print(output.refined)

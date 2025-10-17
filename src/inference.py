from schema import Seq2SeqKwInferConfig
from models import Seq2SeqKeywordGenerator

if __name__ == "__main__":
    # Point to your best checkpoint or final saved model directory (must contain model & tokenizer files)
    cfg = Seq2SeqKwInferConfig(
        model_dir="runs/flan_t5_kw_cuda/checkpoint-500",  # or "runs/flan_t5_kw_cuda" if you saved final there
        # IMPORTANT: this must match whatever you used in training
        prefix="Extract keywords:\n\n",   # e.g., "Extract Keywords\n\nDOCUMENT:\n" — keep it identical
        sep="; ",
        device_preference=None,           
        torch_dtype=None,                 
        max_new_tokens=64,
        num_beams=4,
        repetition_penalty=1.05,
        truncate_source_to=2048,
        batch_size=8,
    )

    kg = Seq2SeqKeywordGenerator(cfg)

    # Single doc
    doc = "FlashAttention reduces IO by tiling attention computation for exact, memory-efficient attention."
    print("Single:", kg.generate(doc))

    # Batch
    docs = [
        "Transformers rely on self-attention to model long-range dependencies in sequences.",
        "Vector databases index embedding vectors to support semantic search and retrieval-augmented generation.",
    ]
    print("Batch:", kg.generate(docs))
"""Compare base Whisper-small vs the QLoRA fine-tuned adapter on held-out audio.

Runs both models on the same 50 held-out Indian-English clips and prints a WER
comparison table with the absolute and relative improvement. This is the Phase 3
completion signal.

    python finetune/evaluate.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ft_utils import load_config, resolve  # noqa: E402

# This file is named evaluate.py, which collides with the HuggingFace `evaluate`
# library. Since the script's own directory sits first on sys.path, drop it once
# ft_utils is imported so `import evaluate` below resolves to the real library.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]


def _transcribe_all(model, processor, dataset, device, batch_size=8) -> list[str]:
    """Greedy-decode every clip's precomputed log-mel features to text."""
    import torch

    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    preds: list[str] = []
    feats = dataset["input_features"]
    for i in range(0, len(feats), batch_size):
        chunk = feats[i : i + batch_size]
        input_features = torch.tensor(chunk, dtype=model.dtype).to(device)
        with torch.no_grad():
            generated = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced,
                max_new_tokens=128,
            )
        preds.extend(processor.tokenizer.batch_decode(generated, skip_special_tokens=True))
    return preds


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import evaluate as hf_evaluate
    from datasets import load_from_disk

    cfg = load_config()
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = resolve(cfg["dataset"]["processed_dir"])
    adapter_dir = resolve(cfg["training"]["output_dir"]) / "adapter"
    if not data_dir.exists():
        raise SystemExit(f"No processed data at {data_dir}. Run prepare_dataset.py.")
    if not adapter_dir.exists():
        raise SystemExit(f"No adapter at {adapter_dir}. Run train.py first.")

    test = load_from_disk(str(data_dir))["test"]
    references = test["reference"]
    print(f"Evaluating on {len(references)} held-out clips (device={device})\n")

    processor = WhisperProcessor.from_pretrained(
        mcfg["name"], language=mcfg["language"], task=mcfg["task"]
    )
    metric = hf_evaluate.load("wer")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- Base model ---
    print("Loading base model…")
    base = WhisperForConditionalGeneration.from_pretrained(
        mcfg["name"], torch_dtype=dtype
    ).to(device)
    base.eval()
    base_preds = _transcribe_all(base, processor, test, device)
    base_wer = 100 * metric.compute(predictions=base_preds, references=references)

    # --- Fine-tuned (base + LoRA adapter) ---
    print("Loading fine-tuned adapter on top of base…")
    ft = WhisperForConditionalGeneration.from_pretrained(
        mcfg["name"], torch_dtype=dtype
    ).to(device)
    ft = PeftModel.from_pretrained(ft, str(adapter_dir)).to(device)
    ft.eval()
    ft_preds = _transcribe_all(ft, processor, test, device)
    ft_wer = 100 * metric.compute(predictions=ft_preds, references=references)

    delta = ft_wer - base_wer
    rel = (delta / base_wer * 100) if base_wer else 0.0

    # Persist the comparison so the backend can serve it at GET /api/wer.
    import json

    result_path = resolve(cfg["training"]["output_dir"]) / "wer_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "available": True,
                "base_wer": round(base_wer, 2),
                "finetuned_wer": round(ft_wer, 2),
                "delta": round(delta, 2),
                "relative": round(rel, 1),
                "model": mcfg["name"],
                "dataset": cfg["dataset"]["name"],
                "samples": len(references),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 44)
    print(f"  {'Model':<18}{'WER':>10}")
    print("-" * 44)
    print(f"  {'Base whisper-small':<18}{base_wer:>9.2f}%")
    print(f"  {'Fine-tuned (QLoRA)':<18}{ft_wer:>9.2f}%")
    print("-" * 44)
    print(f"  {'Delta':<18}{delta:>+9.2f}%  ({rel:+.1f}% relative)")
    print("=" * 44)

    # A couple of side-by-side examples for the notes / demo.
    print("\nExamples (reference | base | fine-tuned):")
    for i in range(min(3, len(references))):
        print(f"  REF : {references[i]}")
        print(f"  BASE: {base_preds[i]}")
        print(f"  FT  : {ft_preds[i]}\n")


if __name__ == "__main__":
    main()

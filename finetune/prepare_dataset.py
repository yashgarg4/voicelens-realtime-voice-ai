"""Download and prepare the Indian-English fine-tuning data.

Streams the ai4bharat/Svarah benchmark (Indian-accented English, parquet),
keeps clips of 2-20 seconds, resamples to 16 kHz, computes Whisper log-mel
input features and tokenised labels, then saves a train/test DatasetDict to disk
for train.py and evaluate.py to consume.

Svarah is "auto"-gated: create a free HuggingFace account, accept the terms at
https://huggingface.co/datasets/ai4bharat/Svarah (approval is automatic), and
export your token as HF_TOKEN before running.

(Why Svarah and not Common Voice 17: Mozilla removed Common Voice from the
HuggingFace Hub in Oct 2025, so mozilla-foundation/common_voice_* now contain no
data. Svarah is purpose-built Indian English and ships as parquet, so it loads
cleanly without a dataset script.)
"""

from __future__ import annotations

import os
import sys

# Make `ft_utils` importable whether run as `python finetune/prepare_dataset.py`
# or from inside the finetune/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ft_utils import load_config, resolve  # noqa: E402


def main() -> None:
    from datasets import Audio, Dataset, DatasetDict, load_dataset
    from transformers import WhisperProcessor

    cfg = load_config()
    dcfg = cfg["dataset"]
    mcfg = cfg["model"]

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "WARNING: HF_TOKEN not set. Svarah is gated; loading will likely "
            "fail. Accept the terms on the dataset page and set HF_TOKEN."
        )

    target_total = int(dcfg["max_samples"]) + int(dcfg["eval_samples"])
    min_s = float(dcfg["min_duration_s"])
    max_s = float(dcfg["max_duration_s"])
    audio_col = dcfg["audio_column"]
    text_col = dcfg["text_column"]

    print(f"Streaming {dcfg['name']} (split={dcfg['split']})…")
    stream = load_dataset(
        dcfg["name"], split=dcfg["split"], streaming=True, token=token
    )
    # Decode audio at 16 kHz on access.
    stream = stream.cast_column(audio_col, Audio(sampling_rate=16_000))

    print(f"Collecting up to {target_total} clips ({min_s}-{max_s}s)…")
    collected: list[dict] = []
    scanned = 0
    for ex in stream:
        scanned += 1
        sentence = (ex.get(text_col) or "").strip()
        if not sentence:
            continue
        # Prefer the provided duration; fall back to computing from the array.
        duration = ex.get("duration")
        audio = ex[audio_col]
        if duration is None:
            duration = len(audio["array"]) / audio["sampling_rate"]
        if not (min_s <= float(duration) <= max_s):
            continue
        collected.append({"array": audio["array"], "sentence": sentence})
        if len(collected) % 50 == 0:
            print(f"  …{len(collected)} kept (scanned {scanned})")
        if len(collected) >= target_total:
            break

    if len(collected) < 10:
        raise SystemExit(
            "Too few clips collected. Check HF_TOKEN / dataset terms and the "
            "duration filter."
        )
    print(f"Collected {len(collected)} clips (scanned {scanned} rows).")

    # Feature extraction + label tokenisation with the Whisper processor.
    processor = WhisperProcessor.from_pretrained(
        mcfg["name"], language=mcfg["language"], task=mcfg["task"]
    )

    def build_features(batch: dict) -> dict:
        feats = processor.feature_extractor(
            batch["array"], sampling_rate=16_000
        ).input_features[0]
        labels = processor.tokenizer(batch["sentence"]).input_ids
        return {
            "input_features": feats,
            "labels": labels,
            "reference": batch["sentence"],
        }

    ds = Dataset.from_list(collected)
    ds = ds.map(build_features, remove_columns=ds.column_names, desc="Extracting log-mels")

    # Hold out the eval split for the WER comparison (cap if we got fewer clips).
    ds = ds.shuffle(seed=cfg["training"]["seed"])
    eval_n = min(int(dcfg["eval_samples"]), max(1, len(ds) // 5))
    splits = DatasetDict(
        {
            "test": ds.select(range(eval_n)),
            "train": ds.select(range(eval_n, len(ds))),
        }
    )

    out_dir = resolve(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(out_dir))
    print(
        f"Saved {len(splits['train'])} train / {len(splits['test'])} test "
        f"examples to {out_dir}"
    )


if __name__ == "__main__":
    main()

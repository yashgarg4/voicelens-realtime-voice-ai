"""QLoRA fine-tuning of Whisper-small on Indian-English audio.

Pipeline:
  1. Load the 4-bit quantized Whisper-small base (this is the "Q" in QLoRA).
  2. Attach LoRA adapters to the attention q_proj / v_proj layers (the "LoRA").
  3. Train ONLY the adapters with a Seq2SeqTrainer; report WER on the held-out
     set. The frozen 4-bit base is never updated.
  4. Save the adapter weights only (a few MB), not the whole model.

Run on a GPU (free Colab T4 is enough) AFTER prepare_dataset.py:
    python finetune/train.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ft_utils import DataCollatorSpeechSeq2SeqWithPadding, load_config, resolve  # noqa: E402

# Running `python finetune/train.py` puts this directory first on sys.path, so a
# bare `import evaluate` would import our sibling finetune/evaluate.py instead of
# the HuggingFace `evaluate` library. ft_utils is already imported above, so drop
# this directory from the path before importing third-party packages.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]


def print_trainable_parameters(model) -> None:
    """The core QLoRA learning moment: how little we actually train."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / total if total else 0.0
    print("=" * 64)
    print(f"  Trainable params : {trainable:,}")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable %      : {pct:.4f}%")
    print("=" * 64)


def main() -> None:
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        BitsAndBytesConfig,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    import evaluate as hf_evaluate

    cfg = load_config()
    mcfg, qcfg, tcfg = cfg["model"], cfg["qlora"], cfg["training"]

    data_dir = resolve(cfg["dataset"]["processed_dir"])
    if not data_dir.exists():
        raise SystemExit(
            f"No processed data at {data_dir}. Run prepare_dataset.py first."
        )
    ds = load_from_disk(str(data_dir))
    print(f"Loaded {len(ds['train'])} train / {len(ds['test'])} test examples")

    processor = WhisperProcessor.from_pretrained(
        mcfg["name"], language=mcfg["language"], task=mcfg["task"]
    )

    # --- 1) 4-bit quantized base -------------------------------------------
    compute_dtype = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        mcfg["name"], quantization_config=bnb_config, device_map="auto"
    )
    # Whisper generation config: let language/task drive decoding, no forced ids.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required while training with gradients

    # --- 2) LoRA adapters ---------------------------------------------------
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=qcfg["lora_r"],
        lora_alpha=qcfg["lora_alpha"],
        target_modules=list(qcfg["target_modules"]),
        lora_dropout=qcfg["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)  # mandatory: shows ~<1% trained

    # --- 3) Trainer + WER metric -------------------------------------------
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    metric = hf_evaluate.load("wer")
    tokenizer = processor.tokenizer

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    output_dir = resolve(tcfg["output_dir"])
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        warmup_steps=tcfg["warmup_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        fp16=tcfg["fp16"],
        predict_with_generate=True,
        generation_max_length=tcfg["generation_max_length"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        logging_steps=tcfg["logging_steps"],
        seed=tcfg["seed"],
        report_to=[],
        # PEFT specifics: keep input_features (Trainer would otherwise drop
        # columns the PeftModel.forward signature doesn't expose) and label them.
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    print("Starting training…")
    trainer.train()

    # --- 4) Save the adapter ONLY ------------------------------------------
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\nSaved LoRA adapter (only) to {adapter_dir}")
    print("Now run: python finetune/evaluate.py")


if __name__ == "__main__":
    main()

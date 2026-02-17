"""
Watty Training Pipeline — The Trivium + Sophia
================================================
Runs all 4 phases sequentially on GPU.
Designed for RTX 4090 (24GB VRAM). Maxes out utilization.

Phase 1: GRAMMAR  — SFT on factual knowledge (lr=2e-4)
Phase 2: LOGIC    — SFT on reasoning patterns (lr=1e-4)
Phase 3: RHETORIC — DPO on style preferences  (beta=0.1)
Phase 4: SOPHIA   — SFT on self-knowledge     (lr=5e-5)

Each phase: curriculum order (simple → complex).
Each phase: builds on the last (LoRA adapters accumulate).

"We are what we repeatedly do. Excellence is not an act, but a habit." — Aristotle
"""

import os
import json
import time
import torch
from pathlib import Path
from datetime import datetime

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_DIR = Path("/workspace/training_data")
OUTPUT_DIR = Path("/workspace/watty_model")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ── Utilities ───────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def gpu_stats():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        util = mem / total * 100
        return f"VRAM: {mem:.1f}/{total:.1f}GB ({util:.0f}%)"
    return "No GPU"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ── Phase 1 & 2 & 4: SFT Training ─────────────────────────

def train_sft(phase_name, data_file, output_dir, learning_rate, num_epochs=1,
              base_model=None, adapter_path=None):
    """Supervised Fine-Tuning with QLoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    log(f"=== PHASE: {phase_name} ===")
    log(f"Data: {data_file}")
    log(f"LR: {learning_rate}, Epochs: {num_epochs}")
    log(f"{gpu_stats()}")

    # Load data
    raw_data = load_jsonl(data_file)
    log(f"Loaded {len(raw_data)} examples")

    # Format for SFT
    def format_messages(example):
        messages = example.get("messages", [])
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        return {"text": "\n".join(text_parts)}

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_messages)
    log(f"Dataset formatted: {len(dataset)} examples")

    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_path = base_model or MODEL_NAME
    log(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load existing adapter if continuing from previous phase
    if adapter_path and Path(adapter_path).exists():
        log(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        # Re-quantize after merge
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # LoRA config — aggressive rank for deep specialization
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    log(f"{gpu_stats()}")

    # Training config — fit in 24GB VRAM with QLoRA
    phase_output = output_dir / phase_name
    training_args = SFTConfig(
        output_dir=str(phase_output),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size = 16
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    log("Starting training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    log(f"Training complete in {elapsed/60:.1f} minutes")
    log(f"{gpu_stats()}")

    # Save adapter
    adapter_out = phase_output / "adapter"
    trainer.save_model(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    log(f"Adapter saved: {adapter_out}")

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return str(adapter_out)


# ── Phase 3: DPO Training ──────────────────────────────────

def train_dpo(data_file, output_dir, adapter_path=None):
    """Direct Preference Optimization — learn Hugo's style."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    log("=== PHASE: RHETORIC (DPO) ===")
    log(f"Data: {data_file}")
    log(f"{gpu_stats()}")

    raw_data = load_jsonl(data_file)
    if len(raw_data) < 5:
        log(f"Only {len(raw_data)} DPO pairs — skipping (need at least 5)")
        return adapter_path

    log(f"Loaded {len(raw_data)} preference pairs")

    # Format for DPO
    formatted = []
    for item in raw_data:
        formatted.append({
            "prompt": item.get("prompt", ""),
            "chosen": item.get("chosen", ""),
            "rejected": item.get("rejected", ""),
        })

    dataset = Dataset.from_list(formatted)

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA for DPO
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    phase_output = output_dir / "rhetoric"
    training_args = DPOConfig(
        output_dir=str(phase_output),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        beta=0.1,
        max_length=768,
        max_prompt_length=384,
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    log("Starting DPO training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    log(f"DPO complete in {elapsed/60:.1f} minutes")

    adapter_out = phase_output / "adapter"
    trainer.save_model(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    log(f"DPO adapter saved: {adapter_out}")

    del model, trainer
    torch.cuda.empty_cache()

    return str(adapter_out)


# ── Inference Test ──────────────────────────────────────────

def test_inference(adapter_path):
    """Quick test of the trained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    log("=== INFERENCE TEST ===")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, adapter_path)

    test_prompts = [
        "What are you?",
        "What do you know about Hugo's projects?",
        "When should you act without asking?",
        "What makes you different from other AI?",
    ]

    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are Watty, Hugo's autonomous AI memory system. Be concise and direct."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"\nQ: {prompt}")
        log(f"A: {response[:300]}")

    del model
    torch.cuda.empty_cache()
    log("\nInference test complete.")


# ── Main Pipeline ───────────────────────────────────────────

def main():
    log("=" * 60)
    log("WATTY TRAINING PIPELINE — THE TRIVIUM + SOPHIA")
    log("=" * 60)
    log(f"Model: {MODEL_NAME}")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB" if torch.cuda.is_available() else "")
    log(f"Data dir: {DATA_DIR}")
    log(f"Output dir: {OUTPUT_DIR}")
    log("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    # Phase 1: GRAMMAR — Learn Hugo's world
    adapter_path = train_sft(
        phase_name="grammar",
        data_file=DATA_DIR / "phase1_grammar.jsonl",
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4,
        num_epochs=1,
    )

    # Phase 2: LOGIC — Learn how Hugo thinks
    adapter_path = train_sft(
        phase_name="logic",
        data_file=DATA_DIR / "phase2_logic.jsonl",
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        num_epochs=1,
    )

    # Phase 3: RHETORIC — Learn Hugo's communication style
    dpo_data = DATA_DIR / "phase3_rhetoric.jsonl"
    if dpo_data.exists():
        adapter_path = train_dpo(
            data_file=dpo_data,
            output_dir=OUTPUT_DIR,
            adapter_path=adapter_path,
        )

    # Phase 4: SOPHIA — Self-knowledge
    adapter_path = train_sft(
        phase_name="sophia",
        data_file=DATA_DIR / "phase4_sophia.jsonl",
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        num_epochs=2,  # Extra epochs — this is the most important phase
    )

    total_time = (time.time() - pipeline_start) / 60
    log("")
    log("=" * 60)
    log(f"ALL PHASES COMPLETE in {total_time:.1f} minutes")
    log(f"Final adapter: {adapter_path}")
    log("=" * 60)

    # Test the trained model
    test_inference(adapter_path)

    log(f"\nTotal pipeline time: {total_time:.1f} minutes")
    log("Watty has been trained. The Trivium is complete.")


if __name__ == "__main__":
    main()

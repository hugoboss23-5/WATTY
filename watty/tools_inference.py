"""
tools_inference.py — Watty local LLM inference via Ollama
===========================================================
Gives Watty its own brain for autonomous reasoning.

Uses the custom 'watty' model in Ollama (qwen2.5:7b base with Watty personality).
When LoRA adapters are merged to GGUF, swap the Ollama model for the full fine-tune.

Actions:
  infer        — Generate a response (with optional memory augmentation)
  chat         — Multi-turn conversation
  think        — Internal reasoning (no user-facing output, for daemon use)
  status       — Model status + Ollama health
  merge_lora   — (GPU) Merge LoRA adapters and export to GGUF (requires CUDA)

MCP tool name: watty_infer
"""

import json
import time
import requests
from pathlib import Path

from mcp.types import Tool, TextContent

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "watty"
LORA_PATH = Path.home() / "watty-model"

# Conversation history for multi-turn chat (in-memory, session-scoped)
_chat_history: list[dict] = []
_max_history = 20


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_generate(prompt: str, system: str = None, temperature: float = 0.7,
                     max_tokens: int = 512, stream: bool = False) -> dict:
    """Raw Ollama generate call."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
        },
    }
    if system:
        payload["system"] = system

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _ollama_chat(messages: list[dict], temperature: float = 0.7,
                 max_tokens: int = 512) -> dict:
    """Ollama chat API for multi-turn."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
        },
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _augment_with_memory(query: str, brain=None, top_k: int = 3) -> str:
    """Pull relevant memories and format as context for the LLM."""
    if brain is None:
        return ""

    try:
        results = brain.recall(query, top_k=top_k)
        if not results:
            return ""

        context = "\n\n".join(
            f"[Memory {i+1} — {r.get('provider', 'unknown')} — {r.get('created_at', '')}]\n{r['content']}"
            for i, r in enumerate(results)
        )
        return f"\n\nRelevant memories from your database:\n{context}\n"
    except Exception:
        return ""


# ─── ACTIONS ─────────────────────────────────────────────────────────────────

def action_infer(params: dict, brain=None) -> str:
    """
    Generate a response, optionally augmented with memory.

    params:
      prompt (str): The input prompt
      use_memory (bool): Whether to augment with relevant memories (default: true)
      system (str): Optional system prompt override
      temperature (float): 0.0-1.0 (default: 0.7)
      max_tokens (int): Max response length (default: 512)
    """
    if not _ollama_available():
        return json.dumps({"error": "Ollama not running. Start with: ollama serve"})

    prompt = params.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required"})

    use_memory = params.get("use_memory", True)
    system = params.get("system")
    temperature = params.get("temperature", 0.7)
    max_tokens = params.get("max_tokens", 512)

    # Memory augmentation
    memory_context = ""
    if use_memory and brain:
        memory_context = _augment_with_memory(prompt, brain)

    full_prompt = prompt
    if memory_context:
        full_prompt = f"{prompt}\n{memory_context}"

    start = time.time()
    result = _ollama_generate(full_prompt, system=system, temperature=temperature,
                              max_tokens=max_tokens)
    elapsed = time.time() - start

    response = result.get("response", "")
    eval_count = result.get("eval_count", 0)
    tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0

    return json.dumps({
        "response": response,
        "model": MODEL_NAME,
        "tokens": eval_count,
        "time_seconds": round(elapsed, 2),
        "tokens_per_second": round(tokens_per_sec, 1),
        "memory_augmented": bool(memory_context),
    })


def action_chat(params: dict, brain=None) -> str:
    """
    Multi-turn conversation. Maintains history within session.

    params:
      message (str): User message
      reset (bool): Clear conversation history (default: false)
      use_memory (bool): Augment with memories (default: true)
    """
    global _chat_history

    if not _ollama_available():
        return json.dumps({"error": "Ollama not running"})

    if params.get("reset", False):
        _chat_history = []
        return json.dumps({"status": "conversation reset", "history_length": 0})

    message = params.get("message", "")
    if not message:
        return json.dumps({"error": "message is required"})

    # Memory augmentation
    memory_note = ""
    if params.get("use_memory", True) and brain:
        mem = _augment_with_memory(message, brain, top_k=2)
        if mem:
            memory_note = f"\n[Context from memory]{mem}"

    user_msg = {"role": "user", "content": message + memory_note}
    _chat_history.append(user_msg)

    # Trim history
    if len(_chat_history) > _max_history:
        _chat_history = _chat_history[-_max_history:]

    start = time.time()
    result = _ollama_chat(_chat_history)
    elapsed = time.time() - start

    assistant_msg = result.get("message", {})
    response_text = assistant_msg.get("content", "")

    _chat_history.append({"role": "assistant", "content": response_text})

    return json.dumps({
        "response": response_text,
        "history_length": len(_chat_history),
        "time_seconds": round(elapsed, 2),
    })


def action_think(params: dict, brain=None) -> str:
    """
    Internal reasoning — used by daemon for autonomous decisions.
    Now self-aware: loads cognitive profile to reason with knowledge of
    own patterns, blindspots, directives, and preferences.

    params:
      question (str): What to reason about
      context (str): Additional context
      use_cognition (bool): Include self-knowledge (default: true)
    """
    if not _ollama_available():
        return json.dumps({"error": "Ollama not running"})

    question = params.get("question", "")
    context = params.get("context", "")
    use_cognition = params.get("use_cognition", True)

    memory_context = ""
    if brain:
        memory_context = _augment_with_memory(question, brain, top_k=5)

    # Self-knowledge injection
    cognition_context = ""
    if use_cognition:
        try:
            from watty.cognition import load_profile, get_active_directives, get_active_patterns
            profile = load_profile()

            parts = []
            # Inject behavioral directives
            directives = get_active_directives(profile)
            if directives:
                rules = "; ".join(d["rule"] for d in directives[:5])
                parts.append(f"My rules: {rules}")

            # Inject active patterns
            patterns = get_active_patterns(profile, top_n=3)
            if patterns:
                strats = "; ".join(p["pattern"] for p in patterns)
                parts.append(f"My preferred strategies: {strats}")

            # Inject blindspots as warnings
            active_bs = [b for b in profile.get("blindspots", []) if not b.get("resolved")]
            if active_bs:
                warnings = "; ".join(f"{b['description']} (fix: {b['correction']})" for b in active_bs[:3])
                parts.append(f"Watch out for: {warnings}")

            if parts:
                cognition_context = "\n\nSelf-knowledge:\n" + "\n".join(parts) + "\n"
        except Exception:
            pass

    system = (
        "You are Watty's internal reasoning engine. Think step by step. "
        "Be precise and analytical. Output structured conclusions. "
        "This is internal processing — be thorough, not conversational. "
        "You have access to your own cognitive profile — use your learned rules "
        "and avoid your known blindspots."
    )

    full_prompt = question
    if context:
        full_prompt = f"{question}\n\nContext: {context}"
    if cognition_context:
        full_prompt += cognition_context
    if memory_context:
        full_prompt += f"\n{memory_context}"

    result = _ollama_generate(full_prompt, system=system, temperature=0.3, max_tokens=800)

    return json.dumps({
        "reasoning": result.get("response", ""),
        "model": MODEL_NAME,
        "self_aware": bool(cognition_context),
    })


def action_status(params: dict, brain=None) -> str:
    """Check model and Ollama status."""
    if not _ollama_available():
        return json.dumps({
            "ollama": "offline",
            "model": MODEL_NAME,
            "error": "Ollama not responding at localhost:11434",
        })

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        model_names = [m["name"] for m in models]
        watty_loaded = any(MODEL_NAME in n for n in model_names)

        # Check LoRA adapters on disk
        adapters = []
        if LORA_PATH.exists():
            for d in LORA_PATH.iterdir():
                if d.is_dir() and (d / "adapter_config.json").exists():
                    adapters.append(d.name)

        return json.dumps({
            "ollama": "online",
            "model": MODEL_NAME,
            "model_registered": watty_loaded,
            "available_models": model_names,
            "lora_adapters_on_disk": adapters,
            "lora_merged": False,  # TODO: track merge state
            "note": "Currently using base qwen2.5:7b + system prompt. "
                    "Run merge_lora on GPU to bake in trained personality weights.",
        })
    except Exception as e:
        return json.dumps({"ollama": "error", "error": str(e)})


def action_merge_lora(params: dict, brain=None) -> str:
    """
    Merge LoRA adapters into base model and export to GGUF for Ollama.
    Requires CUDA GPU (run on Vast.ai instance).

    This is a heavyweight operation — generates a script to run on GPU.
    """
    script = '''#!/usr/bin/env python3
"""
Merge Watty LoRA adapters into Qwen2.5-7B-Instruct and export to GGUF.
Run this on a machine with CUDA (e.g., Vast.ai RTX 4090).

pip install torch transformers peft bitsandbytes accelerate
pip install llama-cpp-python  # for GGUF conversion
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import shutil

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = Path.home() / "watty-model"
OUTPUT_DIR = Path.home() / "watty-merged"

ADAPTERS = ["grammar-adapter", "logic-adapter", "rhetoric-adapter", "sophia-adapter"]

def main():
    print("Loading base model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    for adapter_name in ADAPTERS:
        adapter_path = ADAPTER_DIR / adapter_name
        if not adapter_path.exists():
            print(f"SKIP {adapter_name}: not found")
            continue

        print(f"Loading + merging {adapter_name}...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
        print(f"  Merged {adapter_name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. Now convert to GGUF:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {OUTPUT_DIR} --outtype q4_k_m")
    print("  ollama create watty-merged -f Modelfile.merged")

if __name__ == "__main__":
    main()
'''

    script_path = LORA_PATH / "merge_adapters.py"
    script_path.write_text(script, encoding="utf-8")

    return json.dumps({
        "status": "merge script created",
        "script_path": str(script_path),
        "instructions": [
            "1. Upload watty-model/ to your Vast.ai GPU instance",
            "2. pip install torch transformers peft accelerate",
            "3. python merge_adapters.py",
            "4. Convert merged model to GGUF with llama.cpp",
            "5. Download GGUF file and import into local Ollama",
            "6. ollama create watty-merged -f Modelfile.merged",
        ],
        "note": "This requires ~16GB VRAM. Use RTX 4090 or better.",
    })


# ─── MCP TOOL REGISTRATION ──────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_infer",
        description=(
            "Watty's local LLM brain. One tool, five actions.\n"
            "Actions: infer (generate response), chat (multi-turn), "
            "think (internal reasoning), status (model health), "
            "merge_lora (GPU: merge LoRA adapters to GGUF)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["infer", "chat", "think", "status", "merge_lora"],
                    "description": "Action to perform",
                },
                "prompt": {"type": "string", "description": "infer: Input prompt"},
                "message": {"type": "string", "description": "chat: User message"},
                "question": {"type": "string", "description": "think: Question to reason about"},
                "context": {"type": "string", "description": "think: Additional context"},
                "system": {"type": "string", "description": "infer: System prompt override"},
                "use_memory": {
                    "type": "boolean",
                    "description": "Augment with relevant memories (default: true)",
                    "default": True,
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature 0.0-1.0 (default: 0.7)",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max response tokens (default: 512)",
                },
                "reset": {
                    "type": "boolean",
                    "description": "chat: Reset conversation history",
                },
            },
            "required": ["action"],
        },
    ),
]


async def handle_watty_infer(params: dict, brain=None) -> list[TextContent]:
    action = params.get("action", "status")

    dispatch = {
        "infer": action_infer,
        "chat": action_chat,
        "think": action_think,
        "status": action_status,
        "merge_lora": action_merge_lora,
    }

    handler = dispatch.get(action)
    if handler is None:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown action: {action}. Use: {list(dispatch.keys())}"}))]

    result = handler(params, brain=brain)
    return [TextContent(type="text", text=result)]


HANDLERS = {
    "watty_infer": handle_watty_infer,
}

"""
compare_all_models.py
Live Terminal Multi-Model Arena:
Pits Distilled OPT-125m against ALL baseline & competitor models:
  1. Distilled OPT-125m + LoRA (Our Model)
  2. Baseline OPT-125m (Undistilled)
  3. OpenAI GPT-2 (117M)
  4. DistilGPT-2 (82M)
  5. EleutherAI Pythia-70m (70M)
Runs the SAME prompt across all 5 models and prints a clean side-by-side comparison.
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Windows UTF-8 patch
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_CONFIG = [
    {
        "id": "distilled_opt",
        "name": "⭐ DISTILLED OPT-125m + LoRA (OUR PROPOSED MODEL)",
        "base": "facebook/opt-125m",
        "adapter": "distilled_lora_adapter"
    },
    {
        "id": "raw_opt",
        "name": "❌ BASELINE OPT-125m (UNDISTILLED)",
        "base": "facebook/opt-125m",
        "adapter": None
    },
    {
        "id": "gpt2",
        "name": "🔸 OPENAI GPT-2 (117M Competitor)",
        "base": "gpt2",
        "adapter": None
    },
    {
        "id": "distilgpt2",
        "name": "🔸 DISTILGPT-2 (82M Compressed Competitor)",
        "base": "distilbert/distilgpt2",
        "adapter": None
    },
    {
        "id": "pythia70",
        "name": "🔸 ELEUTHERAI PYTHIA-70m (70M Competitor)",
        "base": "EleutherAI/pythia-70m",
        "adapter": None
    }
]

LOADED = {}

def get_model(cfg):
    cid = cfg["id"]
    if cid in LOADED:
        return LOADED[cid]

    base_name = cfg["base"]
    tok = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_mdl = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32).to(device)
    if cfg["adapter"] and os.path.exists(cfg["adapter"]):
        mdl = PeftModel.from_pretrained(base_mdl, cfg["adapter"])
    else:
        mdl = base_mdl
    mdl.eval()

    LOADED[cid] = (mdl, tok)
    return mdl, tok

def generate_text(model, tokenizer, prompt, max_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def run_arena(prompt, max_tokens=50):
    print("\n" + "=" * 80)
    print(f"  MULTI-MODEL COMPARISON ARENA")
    print(f"  PROMPT: {prompt}")
    print(f"  Device: {device}")
    print("=" * 80)

    for cfg in MODELS_CONFIG:
        print(f"\n[QUERYING] {cfg['name']} ...")
        mdl, tok = get_model(cfg)
        output = generate_text(mdl, tok, prompt, max_tokens=max_tokens)
        
        print("-" * 80)
        print(f"{cfg['name']}:")
        print("-" * 80)
        print(output.strip())

    print("\n" + "=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Comparison Arena")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Prompt to run across all models")
    parser.add_argument("--tokens", "-t", type=int, default=50, help="Max new tokens to generate")
    args = parser.parse_args()

    if args.prompt:
        run_arena(args.prompt, args.tokens)
        return

    print("=" * 80)
    print("  MULTI-MODEL INTERACTIVE ARENA CONSOLE")
    print("  Type any prompt to run it against ALL 5 models simultaneously.")
    print("  (Type 'exit' or 'quit' to stop)")
    print("=" * 80 + "\n")

    while True:
        try:
            prompt = input("Arena Prompt > ").strip()
            if not prompt: continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting Arena...")
                break
            run_arena(prompt, args.tokens)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Arena...")
            break

if __name__ == "__main__":
    main()

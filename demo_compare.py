"""
demo_compare.py
Direct Side-by-Side Live Demo: Baseline OPT-125m vs Distilled OPT-125m (+ LoRA).
Shows both answers simultaneously with side-by-side formatting.
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

MODEL_NAME = "facebook/opt-125m"
ADAPTER_PATH = "distilled_lora_adapter"

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  LOADING DEMO MODELS: BASELINE OPT-125m vs DISTILLED OPT-125m")
    print(f"  Device: {device}")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[1/2] Loading Baseline (Undistilled) OPT-125m ...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    base_model.eval()

    print("[2/2] Loading Distilled OPT-125m (+ MiniLLM LoRA Adapter) ...")
    raw_for_lora = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    if os.path.exists(ADAPTER_PATH):
        distilled_model = PeftModel.from_pretrained(raw_for_lora, ADAPTER_PATH)
    else:
        print(f"[WARN] {ADAPTER_PATH} not found! Using raw model.")
        distilled_model = raw_for_lora
    distilled_model.eval()

    print("\n[OK] Both models loaded and ready for live side-by-side demo!\n")
    return tokenizer, base_model, distilled_model, device

def generate(model, tokenizer, prompt, device, max_tokens=60):
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

def run_comparison(tokenizer, base_model, distilled_model, device, prompt, max_tokens=60):
    print("\n" + "=" * 75)
    print(f"  PROMPT: {prompt}")
    print("=" * 75)

    print("\n>>> Generating from BASELINE OPT-125m (Zero-shot / No KD) ...")
    base_out = generate(base_model, tokenizer, prompt, device, max_tokens)

    print(">>> Generating from DISTILLED OPT-125m (+ LoRA / MiniLLM) ...")
    dist_out = generate(distilled_model, tokenizer, prompt, device, max_tokens)

    print("\n" + "-" * 75)
    print("❌ BASELINE OPT-125m (UNDISTILLED):")
    print("-" * 75)
    print(base_out.strip())

    print("\n" + "=" * 75)
    print("✅ DISTILLED OPT-125m + LoRA (OUR PROPOSED MODEL):")
    print("=" * 75)
    print(dist_out.strip())
    print("\n" + "=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Live Side-by-Side Demo: Baseline vs Distilled")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="One-off prompt to run")
    parser.add_argument("--tokens", "-t", type=int, default=60, help="Max new tokens")
    args = parser.parse_args()

    tokenizer, base_model, distilled_model, device = load_models()

    if args.prompt:
        run_comparison(tokenizer, base_model, distilled_model, device, args.prompt, args.tokens)
        return

    print("Type your prompt and press Enter to see side-by-side outputs.")
    print("Sample prompts to try:")
    print("  1. Question: What is knowledge distillation in machine learning? Answer:")
    print("  2. Question: Can metformin cause hypoglycemia? Answer:")
    print("  3. Question: Explain how vaccines work. Answer:")
    print("(Type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            prompt = input("Demo Prompt > ").strip()
            if not prompt: continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting demo...")
                break
            run_comparison(tokenizer, base_model, distilled_model, device, prompt, args.tokens)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo...")
            break

if __name__ == "__main__":
    main()

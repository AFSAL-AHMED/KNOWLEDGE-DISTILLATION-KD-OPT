1"""
chat.py
Interactive CLI to test and chat with the distilled model or provide one-off prompts.
Supports:
  1. Distilled OPT-125m (+ trained LoRA adapter) [Default]
  2. Baseline / Raw OPT-125m (for side-by-side comparison)
  3. Distilled GPT-2 (via compare.py Seq-KD)
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def load_distilled_opt(adapter_path="distilled_lora_adapter", base_model="facebook/opt-125m"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading base model '{base_model}' on {device} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32).to(device)
    
    if os.path.exists(adapter_path):
        print(f"[INFO] Attaching distilled LoRA adapter from '{adapter_path}' ...")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        print(f"[WARN] Adapter path '{adapter_path}' not found! Using raw base model.")
        model = base
        
    model.eval()
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Chat with your Distilled Model")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="One-off prompt to run")
    parser.add_argument("--adapter", "-a", type=str, default="distilled_lora_adapter", help="Path to LoRA adapter")
    parser.add_argument("--tokens", "-t", type=int, default=80, help="Max new tokens to generate")
    args = parser.parse_args()

    model, tokenizer, device = load_distilled_opt(adapter_path=args.adapter)

    print("\n" + "=" * 70)
    print("  DISTILLED MODEL INTERACTIVE INFERENCE CONSOLE")
    print("  Model: facebook/opt-125m + MiniLLM Reverse-KL LoRA Adapter")
    print("=" * 70)

    # Mode 1: Single command-line prompt
    if args.prompt:
        print(f"\n[PROMPT]: {args.prompt}")
        resp = generate_response(model, tokenizer, args.prompt, device, max_new_tokens=args.tokens)
        print(f"\n[OUTPUT]:\n{resp}\n")
        return

    # Mode 2: Interactive loop
    print("\nType your prompt and press Enter. (Type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break

            response = generate_response(model, tokenizer, user_input, device, max_new_tokens=args.tokens)
            print(f"\nModel >\n{response}\n" + "-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()

"""
plot_results.py
Generates presentation-grade comparison charts for Knowledge Distillation benchmarks.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations(json_path="results_comparison.json", output_dir="charts"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(json_path):
        print(f"[WARN] {json_path} not found. Cannot plot charts.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = list(data.keys())
    if not models:
        print("[WARN] No models found in data.")
        return

    # Clean display labels
    clean_labels = [m.split(":")[0].strip() if ":" in m else m for m in models]

    # Extract Metrics
    dolly_ppl = [data[m].get("DollyEval", {}).get("ppl", 0) for m in models]
    selfinst_ppl = [data[m].get("SelfInst", {}).get("ppl", 0) for m in models]
    avg_ppl = [(d + s) / 2 for d, s in zip(dolly_ppl, selfinst_ppl)]
    
    rouges = []
    for m in models:
        r1 = data[m].get("DollyEval", {}).get("rouge_l") or 0.0
        r2 = data[m].get("SelfInst", {}).get("rouge_l") or 0.0
        rouges.append((r1 + r2) / 2)

    # 1. Perplexity Comparison Chart (Lower is Better)
    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(clean_labels))
    width = 0.35

    colors = ['#1f77b4' if 'OPT-125m' in l else '#aec7e8' for l in models]

    bars = plt.bar(x, avg_ppl, width=0.5, color=colors, edgecolor='black', linewidth=0.8)
    plt.ylabel('Average Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
    plt.title('Benchmark Perplexity Across Student Models (DollyEval + SelfInst)', fontsize=14, fontweight='bold')
    plt.xticks(x, clean_labels, rotation=15, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    ppl_chart_path = os.path.join(output_dir, "perplexity_comparison.png")
    plt.savefig(ppl_chart_path)
    plt.close()
    print(f"[OK] Saved Perplexity Chart -> {ppl_chart_path}")

    # 2. ROUGE-L Comparison Chart (Higher is Better)
    if any(r > 0 for r in rouges):
        plt.figure(figsize=(10, 5), dpi=300)
        bars = plt.bar(x, rouges, width=0.5, color=['#2ca02c' if 'OPT-125m' in l else '#98df8a' for l in models],
                       edgecolor='black', linewidth=0.8)
        plt.ylabel('Average ROUGE-L Score (Higher is Better)', fontsize=12, fontweight='bold')
        plt.title('Instruction-Following Quality (ROUGE-L Score vs Ground Truth)', fontsize=14, fontweight='bold')
        plt.xticks(x, clean_labels, rotation=15, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        rouge_chart_path = os.path.join(output_dir, "rouge_l_comparison.png")
        plt.savefig(rouge_chart_path)
        plt.close()
        print(f"[OK] Saved ROUGE-L Chart -> {rouge_chart_path}")

if __name__ == "__main__":
    generate_visualizations()

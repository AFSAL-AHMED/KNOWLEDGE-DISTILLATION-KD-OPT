# Knowledge Distillation Benchmark & Model Comparison Report

Generated on: 2026-10-06

### Benchmark Performance Across Evaluated Models

| Model Configuration | DollyEval PPL | SelfInst PPL | VicunaEval PPL | Avg PPL | Avg ROUGE-L | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A5: Pythia-160m (Seq-KD)** **(BEST)** | 31.95 | 39.44 | 43.74 | **38.38** | **0.0687** | #1 |
| **A1: OPT-125m (Token-KD)** | 28.54 | 41.88 | 47.10 | **39.17** | **0.1045** | #2 |
| **B1: OPT-125m + OPT-350m Teacher** | 28.54 | 41.88 | 47.10 | **39.17** | **0.1045** | #3 |
| **B2: OPT-125m + GPT2-Med Teacher** | 30.21 | 43.39 | 44.95 | **39.52** | **0.0968** | #4 |
| **A2: GPT-2 (Seq-KD)** | 38.10 | 54.20 | 45.49 | **45.93** | **0.0758** | #5 |
| **A3: DistilGPT2 (Seq-KD)** | 48.78 | 66.12 | 61.68 | **58.86** | **0.0747** | #6 |
| **A4: Pythia-70m (Seq-KD)** | 56.04 | 63.67 | 85.14 | **68.28** | **0.0564** | #7 |


*Note: Lower PPL indicates higher model prediction confidence; higher ROUGE-L indicates closer agreement with ground-truth references.*

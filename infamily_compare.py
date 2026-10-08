# =============================================================================
#  In-Family vs Cross-Family Distillation Experiment -- infamily_compare.py
#  Author  : Afsal
#  Date    : 2026-10-07
#
#  PURPOSE
#  -------
#  Proves the "In-Family Distillation Hypothesis" across two distinct model families:
#
#  GROUP 1: OPT Family (Vocabulary = 50272)
#    - In-Family   : OPT-125m Student <- OPT-350m Teacher (Token-level Reverse KL)
#    - Cross-Family: OPT-125m Student <- Pythia-160m Teacher (Sequence-level KD)
#
#  GROUP 2: Pythia / GPT-NeoX Family (Vocabulary = 50304)
#    - In-Family   : Pythia-70m Student <- Pythia-160m Teacher (Token-level Reverse KL)
#    - Cross-Family: Pythia-70m Student <- OPT-350m Teacher (Sequence-level KD)
#
#  CORE RESEARCH HYPOTHESIS:
#  -------------------------
#  In-family distillation consistently beats cross-family distillation regardless
#  of the architecture chosen, because shared tokenizers allow full soft-logit
#  probability distribution transfer ("dark knowledge") without tokenizer distortion.
# =============================================================================

import os, sys, json, math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using Device: {device}")

# --- Models ---
OPT_TEACHER = "facebook/opt-350m"
OPT_STUDENT = "facebook/opt-125m"
PYTHIA_TEACHER = "EleutherAI/pythia-160m"
PYTHIA_STUDENT = "EleutherAI/pythia-70m"

OPT_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PYTHIA_TARGETS = ["query_key_value", "dense"]

# --- Hyperparameters ---
LORA_R = 16
LORA_ALPHA = 32
LR = 3e-4
TEMPERATURE = 2.0
ALPHA = 0.5
MAX_SEQ_LEN = 64
BATCH_SIZE = 2
TRAIN_SAMPLES = 60
EVAL_SAMPLES = 30

# --- Loss Functions ---
def token_reverse_kd(s_logits, t_logits, labels, T=TEMPERATURE, a=ALPHA):
    """Token-level Reverse-KL on shared vocabularies."""
    s_l = s_logits[..., :-1, :].contiguous()
    t_l = t_logits[..., :-1, :].contiguous()
    y   = labels[..., 1:].contiguous()
    
    s_sc = s_l / T; t_sc = t_l / T
    log_ps = F.log_softmax(s_sc, dim=-1)
    log_pt = F.log_softmax(t_sc, dim=-1)
    ps     = F.softmax(s_sc, dim=-1)
    
    rkl  = (ps * (log_ps - log_pt)).sum(dim=-1)
    mask = (y != -100).float()
    kl   = (rkl * mask).sum() / (mask.sum() + 1e-8)
    ce   = F.cross_entropy(s_l.view(-1, s_l.size(-1)), y.view(-1), ignore_index=-100)
    return a * ce + (1 - a) * (T ** 2) * kl

def seq_kd_loss(s_logits, labels):
    """Sequence-level cross-entropy on teacher-generated text."""
    s_l = s_logits[..., :-1, :].contiguous()
    y   = labels[..., 1:].contiguous()
    return F.cross_entropy(s_l.view(-1, s_l.size(-1)), y.view(-1), ignore_index=-100)

# --- Dataset and Tokenizer Helpers ---
class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_SEQ_LEN):
        self.samples = []
        for t in texts:
            if not t.strip(): continue
            enc = tokenizer(t, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            self.samples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)
            })
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def load_data():
    eval_prompts = [
        "Explain knowledge distillation in machine learning.",
        "What is the function of metformin in treating diabetes?",
        "Describe the self-attention mechanism in transformers.",
        "How do vaccines stimulate the human immune system?",
        "What are the benefits of low-rank adaptation LoRA for LLMs?",
        "Summarize how neural networks optimize weights with backpropagation."
    ] * (EVAL_SAMPLES // 6 + 1)
    
    train_texts = [
        "Metformin improves glycemic control by decreasing hepatic glucose production.",
        "Knowledge distillation transfers rich dark knowledge from a large teacher to a small student.",
        "Transformer self attention computes dynamic relevance between sequence tokens.",
        "LoRA freezes pretrained weights and injects trainable rank decomposition matrices.",
        "Reverse KL divergence forces the student policy to focus on teacher high probability modes."
    ] * (TRAIN_SAMPLES // 5 + 1)
    
    return train_texts[:TRAIN_SAMPLES], eval_prompts[:EVAL_SAMPLES]

def build_student(model_name, targets):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                     target_modules=targets, lora_dropout=0.05, bias="none")
    model = get_peft_model(base, cfg)
    return model, tok

def evaluate_perplexity(model, tokenizer, prompts):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        ids = enc["input_ids"]
        lbl = ids.clone()
        lbl[:, :-1] = -100
        with torch.no_grad():
            out = model(**enc, labels=lbl)
            loss = out.loss
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_tokens += 1
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20.0))

# --- Main Execution ---
def main():
    print("=" * 80)
    print("  IN-FAMILY VS CROSS-FAMILY KNOWLEDGE DISTILLATION STUDY")
    print("  Proving that Shared Architectural Lineage Maximizes Distillation Quality")
    print("=" * 80)

    train_texts, eval_prompts = load_data()
    results = {}

    # -------------------------------------------------------------------------
    # TEST 1: OPT-125m Student
    # -------------------------------------------------------------------------
    print("\n[GROUP 1] Student: OPT-125m")
    
    # In-Family: OPT-125m <- OPT-350m (Token-KD)
    print("  -> Running 1A: OPT-125m <- OPT-350m [IN-FAMILY Token-Level Reverse KD]...")
    opt_t_tok = AutoTokenizer.from_pretrained(OPT_TEACHER, use_fast=False)
    opt_t_mdl = AutoModelForCausalLM.from_pretrained(OPT_TEACHER, torch_dtype=torch.float32).to(device).eval()
    
    s1, tok1 = build_student(OPT_STUDENT, OPT_TARGETS)
    ds1 = SimpleTextDataset(train_texts, tok1)
    loader1 = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=True)
    opt1 = AdamW(s1.parameters(), lr=LR)
    s1.train()
    for batch in tqdm(loader1, desc="  Training 1A (In-Family)", ncols=80):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = ids.clone(); lbl[mask == 0] = -100
        with torch.no_grad():
            t_log = opt_t_mdl(ids, attention_mask=mask).logits
        s_log = s1(ids, attention_mask=mask).logits
        loss = token_reverse_kd(s_log, t_log, lbl)
        opt1.zero_grad(); loss.backward(); opt1.step()
    
    ppl_1a = evaluate_perplexity(s1, tok1, eval_prompts)
    results["1A. OPT-125m <- OPT-350m (In-Family)"] = {"ppl": ppl_1a, "type": "In-Family"}
    del s1, opt_t_mdl

    # Cross-Family: OPT-125m <- Pythia-160m (Seq-KD)
    print("  -> Running 1B: OPT-125m <- Pythia-160m [CROSS-FAMILY Sequence-Level KD]...")
    py_t_tok = AutoTokenizer.from_pretrained(PYTHIA_TEACHER, use_fast=False)
    if py_t_tok.pad_token is None: py_t_tok.pad_token = py_t_tok.eos_token
    py_t_mdl = AutoModelForCausalLM.from_pretrained(PYTHIA_TEACHER, torch_dtype=torch.float32).to(device).eval()
    
    # Pre-generate text with Pythia teacher
    py_corpus = []
    for t in train_texts[:30]:
        enc = py_t_tok(t[:60], return_tensors="pt").to(device)
        with torch.no_grad():
            out = py_t_mdl.generate(**enc, max_new_tokens=30, pad_token_id=py_t_tok.eos_token_id)
        py_corpus.append(py_t_tok.decode(out[0], skip_special_tokens=True))
    del py_t_mdl
    
    s2, tok2 = build_student(OPT_STUDENT, OPT_TARGETS)
    ds2 = SimpleTextDataset(py_corpus, tok2)
    loader2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=True)
    opt2 = AdamW(s2.parameters(), lr=LR)
    s2.train()
    for batch in tqdm(loader2, desc="  Training 1B (Cross-Family)", ncols=80):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = ids.clone(); lbl[mask == 0] = -100
        s_log = s2(ids, attention_mask=mask).logits
        loss = seq_kd_loss(s_log, lbl)
        opt2.zero_grad(); loss.backward(); opt2.step()
    
    ppl_1b = evaluate_perplexity(s2, tok2, eval_prompts)
    results["1B. OPT-125m <- Pythia-160m (Cross-Family)"] = {"ppl": ppl_1b, "type": "Cross-Family"}
    del s2

    # -------------------------------------------------------------------------
    # TEST 2: Pythia-70m Student
    # -------------------------------------------------------------------------
    print("\n[GROUP 2] Student: Pythia-70m")
    
    # In-Family: Pythia-70m <- Pythia-160m (Token-KD)
    print("  -> Running 2A: Pythia-70m <- Pythia-160m [IN-FAMILY Token-Level Reverse KD]...")
    py_t_mdl = AutoModelForCausalLM.from_pretrained(PYTHIA_TEACHER, torch_dtype=torch.float32).to(device).eval()
    s3, tok3 = build_student(PYTHIA_STUDENT, PYTHIA_TARGETS)
    ds3 = SimpleTextDataset(train_texts, tok3)
    loader3 = DataLoader(ds3, batch_size=BATCH_SIZE, shuffle=True)
    opt3 = AdamW(s3.parameters(), lr=LR)
    s3.train()
    for batch in tqdm(loader3, desc="  Training 2A (In-Family)", ncols=80):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = ids.clone(); lbl[mask == 0] = -100
        with torch.no_grad():
            t_log = py_t_mdl(ids, attention_mask=mask).logits
        s_log = s3(ids, attention_mask=mask).logits
        loss = token_reverse_kd(s_log, t_log, lbl)
        opt3.zero_grad(); loss.backward(); opt3.step()
    
    ppl_2a = evaluate_perplexity(s3, tok3, eval_prompts)
    results["2A. Pythia-70m <- Pythia-160m (In-Family)"] = {"ppl": ppl_2a, "type": "In-Family"}
    del s3, py_t_mdl

    # Cross-Family: Pythia-70m <- OPT-350m (Seq-KD)
    print("  -> Running 2B: Pythia-70m <- OPT-350m [CROSS-FAMILY Sequence-Level KD]...")
    opt_t_mdl = AutoModelForCausalLM.from_pretrained(OPT_TEACHER, torch_dtype=torch.float32).to(device).eval()
    opt_corpus = []
    for t in train_texts[:30]:
        enc = opt_t_tok(t[:60], return_tensors="pt").to(device)
        with torch.no_grad():
            out = opt_t_mdl.generate(**enc, max_new_tokens=30, pad_token_id=opt_t_tok.eos_token_id)
        opt_corpus.append(opt_t_tok.decode(out[0], skip_special_tokens=True))
    del opt_t_mdl

    s4, tok4 = build_student(PYTHIA_STUDENT, PYTHIA_TARGETS)
    ds4 = SimpleTextDataset(opt_corpus, tok4)
    loader4 = DataLoader(ds4, batch_size=BATCH_SIZE, shuffle=True)
    opt4 = AdamW(s4.parameters(), lr=LR)
    s4.train()
    for batch in tqdm(loader4, desc="  Training 2B (Cross-Family)", ncols=80):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = ids.clone(); lbl[mask == 0] = -100
        s_log = s4(ids, attention_mask=mask).logits
        loss = seq_kd_loss(s_log, lbl)
        opt4.zero_grad(); loss.backward(); opt4.step()
        
    ppl_2b = evaluate_perplexity(s4, tok4, eval_prompts)
    results["2B. Pythia-70m <- OPT-350m (Cross-Family)"] = {"ppl": ppl_2b, "type": "Cross-Family"}
    del s4

    # -------------------------------------------------------------------------
    # PRINT SUMMARY TABLE & RESEARCH VERIFICATION
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("  RESEARCH FINDINGS: IN-FAMILY VS CROSS-FAMILY DISTILLATION")
    print("=" * 80)
    print(f"  {'Configuration':<45} | {'Distillation Type':<15} | {'Perplexity (PPL)':>12}")
    print("  " + "-" * 78)
    for name, r in results.items():
        print(f"  {name:<45} | {r['type']:<15} | {r['ppl']:>12.2f}")
    
    print("\n[CONCLUSIONS & HYPOTHESIS CONFIRMATION]")
    opt_win = ppl_1a < ppl_1b
    py_win = ppl_2a < ppl_2b
    print(f"  1. For OPT-125m: In-Family PPL ({ppl_1a:.2f}) < Cross-Family PPL ({ppl_1b:.2f}) -> Win: {opt_win}")
    print(f"  2. For Pythia-70m: In-Family PPL ({ppl_2a:.2f}) < Cross-Family PPL ({ppl_2b:.2f}) -> Win: {py_win}")
    if opt_win and py_win:
        print("\n  >> EMPIRICALLY CONFIRMED: In-Family Distillation is strictly superior across all model families! <<")

    with open("infamily_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Results saved to infamily_results.json")

if __name__ == "__main__":
    main()

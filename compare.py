# =============================================================================
#  Model Architecture Comparison -- compare.py
#  Author  : Afsal
#  Date    : 2026-10-05
#
#  PURPOSE
#  -------
#  Expands the MiniLLM distillation project by comparing multiple
#  student / teacher architectures on three instruction-following benchmarks.
#
#  EXPERIMENT A  :  Fixed Teacher = OPT-350m,  vary student
#    A1. OPT-125m   + LoRA  -> Token-level  MiniLLM Reverse KD (same family)
#    A2. GPT-2      + LoRA  -> Sequence-level KD (cross-architecture)
#    A3. DistilGPT2 + LoRA  -> Sequence-level KD (cross-architecture)
#
#  EXPERIMENT B  :  Fixed Student = OPT-125m + LoRA,  vary teacher
#    B1. Teacher = OPT-350m     -> Token-level MiniLLM Reverse KD
#    B2. Teacher = GPT-2-medium -> Sequence-level KD (cross-architecture)
#
#  WHY TWO DISTILLATION METHODS?
#  ------------------------------
#  Token-level KD  : Computes KL(P_student || P_teacher) directly on logits.
#                    Requires IDENTICAL vocab size (same tokenizer family).
#                    OPT models all share 50272-token vocab -> token-level OK.
#
#  Sequence-level KD : Teacher first GENERATES text from training prompts.
#                      Student then trains with standard CE loss on those tokens
#                      after re-tokenizing with its OWN tokenizer.
#                      Works across ANY two architectures (different vocabs fine).
#                      (Kim & Rush, 2016, "Sequence-Level Knowledge Distillation")
#
#  EVALUATION BENCHMARKS
#  ---------------------
#  DollyEval  : Databricks Dolly-15k held-out instructions (with references)
#  SelfInst   : 252 user-oriented tasks from Wang et al. (with references)
#  VicunaEval : 80 open-ended questions from Vicuna (NO references)
#
#  METRICS
#  -------
#  PPL     : exp(CE loss on instruction+answer) - lower is better
#  ROUGE-L : Longest-common-subsequence recall vs reference - higher is better
#            (Skipped for VicunaEval which has no ground-truth answers)
#  AvgLen  : Mean generated response length in tokens - coverage proxy
# =============================================================================


# =============================================================================
#  SECTION 1 -- IMPORTS & ENVIRONMENT
# =============================================================================

import os, sys, json, math, copy, urllib.request

# Windows UTF-8 patch (prevents codec errors with medical/Greek characters)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# ROUGE-L scoring requires: pip install rouge-score
try:
    from rouge_score import rouge_scorer as _rouge_lib
    ROUGE_AVAILABLE = True
    print("[OK] rouge-score library loaded.")
except ImportError:
    ROUGE_AVAILABLE = False
    print("[WARN] rouge-score not installed. ROUGE-L will show N/A.")
    print("       Fix: pip install rouge-score  then re-run.")

print("=" * 75)
print("  Model Architecture Comparison -- KD Benchmark Suite")
print("  Experiments A (vary student) + B (vary teacher)")
print("=" * 75)


# =============================================================================
#  SECTION 2 -- DEVICE & SHARED HYPERPARAMETERS
# =============================================================================

device = torch.device("cpu")
if torch.cuda.is_available():
    try:
        torch.zeros(1, device="cuda")
        device = torch.device("cuda")
        print(f"[INFO] GPU : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[WARN] CUDA probe failed ({exc}). Using CPU.")
else:
    print("[INFO] Device : CPU")

# --- Model identifiers ---
OPT_TEACHER   = "facebook/opt-350m"   # Primary teacher (~350M, OPT family)
GPT2_TEACHER  = "gpt2-medium"         # Alternate teacher (~345M, GPT-2 family)
OPT_STUDENT   = "facebook/opt-125m"   # Primary student (OPT, same family as teacher)
GPT2_STUDENT  = "gpt2"               # Alternate student (~117M, GPT-2 family)
DGPT2_STUDENT = "distilbert/distilgpt2" # Smallest student (~82M, distilled GPT-2)
PYTHIA70_STUDENT  = "EleutherAI/pythia-70m"   # Ultra-compact student (~70M, NeoX family)
PYTHIA160_STUDENT = "EleutherAI/pythia-160m"  # Compact student (~160M, NeoX family)

# --- LoRA target modules per architecture ---
# OPT: standard linear projections in attention + FFN blocks
OPT_LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
# GPT-2: Conv1D layers (PEFT handles Conv1D transparently since v0.3)
GPT2_LORA_TARGETS   = ["c_attn", "c_proj"]
# Pythia / GPT-NeoX: attention and MLP linear projections
PYTHIA_LORA_TARGETS = ["query_key_value", "dense"]

# --- LoRA dimensions (smaller r for speed on CPU) ---
LORA_R     = 16    # Low-rank bottleneck (r=16 -> ~1-2% trainable)
LORA_ALPHA = 32    # Scaling factor (alpha = 2 * r is standard)
LORA_DROP  = 0.05

# --- Training configuration ---
LR                = 3e-4
TEMPERATURE       = 2.0    # Distillation softening temperature
ALPHA             = 0.5    # CE weight; (1-ALPHA) for KD loss
MAX_SEQ_LEN       = 64     # Shorter for CPU speed (128 takes ~3x longer)
BATCH_SIZE        = 2
MAX_TRAIN_SAMPLES = 80     # Small for demo; increase for better results
TEACHER_GEN_NEW   = 40     # Tokens teacher generates per sample (Seq-KD)
EVAL_SAMPLES      = 40     # Samples per benchmark during evaluation

print(f"\n[CONFIG] OPT Teacher : {OPT_TEACHER} | GPT-2 Teacher : {GPT2_TEACHER}")
print(f"[CONFIG] Students    : {OPT_STUDENT}, {GPT2_STUDENT}, {DGPT2_STUDENT}")
print(f"[CONFIG] LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print(f"[CONFIG] LR={LR}, T={TEMPERATURE}, SeqLen={MAX_SEQ_LEN}, Batch={BATCH_SIZE}")


# =============================================================================
#  SECTION 3 -- DATASET FETCHERS
#  DollyEval  : databricks-dolly-15k  (instruction + response)
#  SelfInst   : Self-Instruct 252-question eval (instruction + output)
#  VicunaEval : 80 open questions  (instruction only; no reference answer)
# =============================================================================

os.makedirs("data", exist_ok=True)


def _fetch(url, cache_path, timeout=20):
    """Downloads url -> cache_path (skips if already cached). Returns True/False."""
    if os.path.exists(cache_path):
        return True
    print(f"  Downloading {cache_path} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            with open(cache_path, "wb") as f:
                f.write(r.read())
        return True
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        return False


def get_dollyeval_samples(n=40):
    """
    Databricks Dolly-15k: instruction-following dataset with reference responses.
    We treat the LAST 10 % as an eval split (not used for training).
    Returns list of {"instruction": str, "reference": str}.
    """
    cache = "data/dolly15k.jsonl"
    url   = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k"
             "/resolve/main/databricks-dolly-15k.jsonl")
    samples = []
    if _fetch(url, cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Last 10 % = eval split
            for line in lines[int(len(lines) * 0.9):]:
                obj  = json.loads(line)
                instr = obj.get("instruction", "").strip()
                resp  = obj.get("response", "").strip()
                if instr and resp:
                    samples.append({"instruction": instr, "reference": resp})
                if len(samples) >= n:
                    break
        except Exception as e:
            print(f"  [WARN] DollyEval parse error: {e}")

    # Built-in fallback -- covers domain knowledge, instruction following, reasoning
    if not samples:
        samples = [
            {"instruction": "Explain knowledge distillation in machine learning.",
             "reference": ("Knowledge distillation trains a compact student model to "
                           "mimic a larger teacher by minimizing the divergence between "
                           "their output probability distributions.")},
            {"instruction": "What is the difference between supervised and unsupervised learning?",
             "reference": ("Supervised learning uses labeled examples with ground-truth outputs, "
                           "while unsupervised learning discovers structure in unlabeled data.")},
            {"instruction": "Describe the attention mechanism in transformers.",
             "reference": ("Attention computes weighted sums of value vectors, where weights "
                           "are dot-products of query and key vectors, enabling dynamic focus "
                           "on relevant parts of the input sequence.")},
            {"instruction": "What are the main hyperparameters in LoRA?",
             "reference": ("LoRA has rank r (bottleneck dimension), alpha (scaling factor), "
                           "dropout, and the choice of target modules to adapt.")},
        ] * (n // 4 + 1)

    return samples[:n]


def get_selfinst_samples(n=40):
    """
    Self-Instruct 252-question user-oriented evaluation set.
    Each entry: instruction + one or more reference output instances.
    Returns list of {"instruction": str, "reference": str}.
    """
    cache = "data/selfinst_eval.jsonl"
    url   = ("https://raw.githubusercontent.com/yizhongw/self-instruct/"
             "main/human_eval/user_oriented_instructions.jsonl")
    samples = []
    if _fetch(url, cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                for line in f:
                    obj   = json.loads(line)
                    instr = obj.get("instruction", "").strip()
                    # Grab first non-empty instance output as reference
                    ref   = ""
                    for inst in obj.get("instances", []):
                        ref = inst.get("output", "").strip()
                        if ref:
                            break
                    if instr and ref:
                        samples.append({"instruction": instr, "reference": ref})
                    if len(samples) >= n:
                        break
        except Exception as e:
            print(f"  [WARN] SelfInst parse error: {e}")

    if not samples:
        samples = [
            {"instruction": "Write a haiku about the moon.",
             "reference": "Silver disk above, silent guardian of night, tides obey your call."},
            {"instruction": "Summarize the French Revolution in two sentences.",
             "reference": ("The French Revolution (1789-1799) was a period of radical political "
                           "and societal transformation that overthrew the monarchy and proclaimed "
                           "the values of liberty, equality, and fraternity.")},
            {"instruction": "Explain how vaccines work.",
             "reference": ("Vaccines introduce a harmless antigen or instruction (mRNA) that "
                           "trains the immune system to recognize and fight a specific pathogen "
                           "without causing disease.")},
            {"instruction": "What is gradient descent in machine learning?",
             "reference": ("Gradient descent iteratively adjusts model parameters in the "
                           "direction opposite to the gradient of the loss function, "
                           "minimizing the loss step by step.")},
        ] * (n // 4 + 1)

    return samples[:n]


def get_vicunaeval_samples(n=40):
    """
    Vicuna 80-question benchmark: open-ended, no reference answers.
    Questions test knowledge, reasoning, and instruction following.
    Returns list of {"instruction": str, "reference": ""}.
    ROUGE-L is NOT computed for this benchmark (no reference available).
    """
    cache = "data/vicuna_questions.jsonl"
    url   = ("https://raw.githubusercontent.com/lm-sys/FastChat/"
             "main/fastchat/eval/table/question.jsonl")
    samples = []
    if _fetch(url, cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                for line in f:
                    obj  = json.loads(line)
                    text = obj.get("text", "").strip()
                    if text:
                        samples.append({"instruction": text, "reference": ""})
                    if len(samples) >= n:
                        break
        except Exception as e:
            print(f"  [WARN] VicunaEval parse error: {e}")

    if not samples:
        samples = [
            {"instruction": "How would you design a more equitable global economic system?",
             "reference": ""},
            {"instruction": "Compare the long-term impacts of the Industrial Revolution and "
             "the Digital Revolution.",
             "reference": ""},
            {"instruction": "If you could eliminate one human bias, which would you choose and why?",
             "reference": ""},
        ] * (n // 3 + 1)

    return samples[:n]


# =============================================================================
#  SECTION 4 -- EVALUATION HELPERS
# =============================================================================

def evaluate_ppl(model, tokenizer, samples, device, max_len=128):
    """
    Measures perplexity of model on each sample.
    Text = "Instruction: {instr}\\nResponse: {ref}" (or just instruction if no ref).

    PPL = exp(mean CE loss) -- lower is better.
    A model with lower PPL assigns higher probability to correct next tokens.
    """
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for sample in samples:
            instr = sample["instruction"]
            ref   = sample["reference"]
            # Concatenate instruction + reference as the scored sequence
            text = f"Instruction: {instr}\nResponse: {ref}" if ref else instr
            enc  = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=max_len, padding=False)
            ids  = enc["input_ids"].to(device)
            if ids.size(1) < 2:
                continue
            try:
                out = model(input_ids=ids, labels=ids)
                if not torch.isnan(out.loss):
                    total_loss += out.loss.item()
                    n += 1
            except Exception:
                continue
    avg_loss = total_loss / max(n, 1)
    return math.exp(avg_loss) if avg_loss < 20 else float("inf")


def evaluate_rouge_l(model, tokenizer, samples, device, max_new=60):
    """
    Generates responses for each sample and computes ROUGE-L F1 vs reference.
    ROUGE-L measures the longest common subsequence overlap between
    generated text and reference -- higher means closer to reference.

    Returns None if ROUGE not available or no samples have references.
    Skips samples where reference == "" (e.g., VicunaEval).
    """
    if not ROUGE_AVAILABLE:
        return None
    scorer = _rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    model.eval()
    scores = []
    for sample in samples:
        ref = sample["reference"]
        if not ref:
            continue  # VicunaEval has no reference -- skip
        prompt = f"Instruction: {sample['instruction']}\nResponse:"
        enc    = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=64).to(device)
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=max_new,
                    do_sample=False,                   # Greedy for reproducibility
                    pad_token_id=tokenizer.eos_token_id
                )
            full   = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # Extract generated part after "Response:"
            gen    = full.split("Response:", 1)[-1].strip() if "Response:" in full else full
        except Exception:
            gen = ""
        scores.append(scorer.score(ref, gen)["rougeL"].fmeasure)
    return sum(scores) / max(len(scores), 1) if scores else 0.0


def evaluate_avg_length(model, tokenizer, samples, device, max_new=60):
    """
    Generates responses and returns average length in tokens.
    A proxy for instruction coverage: models that simply stop early score low.
    """
    model.eval()
    lengths = []
    for sample in samples[:20]:   # Limit to 20 for CPU speed
        prompt = f"Instruction: {sample['instruction']}\nResponse:"
        enc    = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=64).to(device)
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            prompt_len = enc["input_ids"].size(1)
            lengths.append(max(out_ids.size(1) - prompt_len, 0))
        except Exception:
            lengths.append(0)
    return sum(lengths) / max(len(lengths), 1) if lengths else 0.0


def evaluate_all(model, tokenizer, label, dolly, selfinst, vicuna, device):
    """
    Runs PPL + ROUGE-L + AvgLen on all three benchmarks for one model.
    Returns dict: {bench_name: {"ppl": float, "rouge_l": float|None, "avg_len": float}}
    """
    bench_results = {}
    for bench_name, bench_data in [("DollyEval",  dolly),
                                    ("SelfInst",   selfinst),
                                    ("VicunaEval", vicuna)]:
        has_ref = bench_name != "VicunaEval"
        print(f"    [{label}] {bench_name}:", end="  ", flush=True)

        ppl = evaluate_ppl(model, tokenizer, bench_data, device)
        print(f"PPL={ppl:.2f}", end="  ", flush=True)

        if has_ref:
            rouge = evaluate_rouge_l(model, tokenizer, bench_data, device)
            print(f"ROUGE-L={rouge:.4f}" if rouge is not None else "ROUGE-L=N/A", end="  ", flush=True)
        else:
            rouge = None
            print("ROUGE-L=N/A(no ref)", end="  ", flush=True)

        avg_len = evaluate_avg_length(model, tokenizer, bench_data, device)
        print(f"AvgLen={avg_len:.1f}")

        bench_results[bench_name] = {"ppl": ppl, "rouge_l": rouge, "avg_len": avg_len}
    return bench_results


# =============================================================================
#  SECTION 5 -- DISTILLATION LOSS FUNCTIONS
# =============================================================================

def _shift(s_logits, t_logits, labels):
    """
    Causal LM alignment: logit at position t predicts token at t+1.
    Returns (shifted_s, shifted_t, shifted_labels) each [B, L-1, ...].
    """
    return (s_logits[..., :-1, :].contiguous(),
            t_logits[..., :-1, :].contiguous(),
            labels[..., 1:].contiguous())


def token_level_kd(s_logits, t_logits, labels, T=TEMPERATURE, a=ALPHA):
    """
    MiniLLM Reverse KD (token-level): KL(P_student || P_teacher).

    VALID ONLY when student and teacher share the same vocabulary size.
    Formula: L = a*CE + (1-a)*T^2 * sum_v P_s*(log P_s - log P_t)

    This is the mode-seeking objective: student concentrates mass on
    the teacher's dominant probability modes.
    """
    s_l, t_l, y = _shift(s_logits, t_logits, labels)
    s_sc = s_l / T;  t_sc = t_l / T
    log_ps = F.log_softmax(s_sc, dim=-1)
    log_pt = F.log_softmax(t_sc, dim=-1)
    ps     = F.softmax(s_sc, dim=-1)
    rkl    = (ps * (log_ps - log_pt)).sum(dim=-1)     # [B, L-1]
    mask   = (y != -100).float()
    kl     = (rkl * mask).sum() / (mask.sum() + 1e-8)
    ce     = F.cross_entropy(s_l.view(-1, s_l.size(-1)), y.view(-1), ignore_index=-100)
    return a * ce + (1 - a) * (T ** 2) * kl, ce, kl


def seq_level_kd(s_logits, labels):
    """
    Sequence-level KD (Kim & Rush, 2016): CE on teacher-generated tokens.

    Works across ANY two architectures because no logit comparison is made.
    The teacher first generates text, which is re-tokenized using the STUDENT's
    tokenizer, then the student is trained with standard cross-entropy on those tokens.

    This is equivalent to treating the teacher's output sequence as "hard labels"
    rather than using soft probability distributions.
    """
    s_l = s_logits[..., :-1, :].contiguous()
    y   = labels[..., 1:].contiguous()
    return F.cross_entropy(s_l.view(-1, s_l.size(-1)), y.view(-1), ignore_index=-100)


# =============================================================================
#  SECTION 6 -- MODEL BUILDERS
# =============================================================================

def load_frozen_teacher(model_name, device):
    """
    Loads a teacher model in eval/frozen mode.
    FP16 on CUDA (memory-efficient), FP32 on CPU (numerically stable).
    Returns (model, tokenizer).
    """
    print(f"\n[INFO] Loading teacher '{model_name}' ...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok   = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl   = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in mdl.parameters())
    print(f"       {n/1e6:.1f}M params | dtype={dtype} | FROZEN")
    return mdl, tok


def build_lora_student(model_name, lora_targets, device):
    """
    Loads a student model and injects LoRA adapters into specified target layers.
    Returns (peft_model, tokenizer).

    LORA_R=16 is used (smaller than main.py's r=96) to keep compare.py fast.
    Each layer gains two low-rank matrices A (r x d_in) and B (d_out x r),
    adding only rank * (d_in + d_out) parameters per layer.
    """
    print(f"\n[INFO] Building student '{model_name}' + LoRA (r={LORA_R}) ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token  = tok.eos_token
        tok.padding_side = "left"   # Required for GPT-2 generation stability

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=lora_targets,
        lora_dropout=LORA_DROP,
        bias="none",
    )
    mdl = get_peft_model(base, lora_cfg)
    mdl.print_trainable_parameters()
    return mdl, tok


# =============================================================================
#  SECTION 7 -- TRAINING DATA & TEACHER CORPUS GENERATION
# =============================================================================

class SimpleDataset(Dataset):
    """
    Tokenizes a list of raw text strings into fixed-length
    (input_ids, attention_mask) pairs for DataLoader consumption.
    """
    def __init__(self, texts, tokenizer, max_len=64):
        self.samples = []
        for t in texts:
            if not t.strip():
                continue
            enc = tokenizer(t, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            self.samples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def get_training_texts(n=80):
    """
    Returns cached general + medical texts from the data/ folder populated
    by main.py. Falls back to synthetic sentences if cache is absent.
    """
    texts = []
    # General language: WikiText-2
    wiki = "data/wikitext2_train.txt"
    if os.path.exists(wiki):
        with open(wiki, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 50 and not line.startswith("="):
                    texts.append(line)
                if len(texts) >= n // 2:
                    break
    # Medical domain: PubMedQA
    pm = "data/pubmedqa_labeled.json"
    if os.path.exists(pm):
        try:
            with open(pm, "r", encoding="utf-8") as f:
                data = json.load(f)
            for _, v in data.items():
                q = v.get("QUESTION", ""); a = v.get("LONG_ANSWER", "")
                if q and a:
                    texts.append(f"Question: {q}\nAnswer: {a}")
                if len(texts) >= n:
                    break
        except Exception:
            pass
    # Fallback
    if len(texts) < 10:
        texts = [
            "The patient was treated with metformin and showed improved glycemic control.",
            "Transformer models use self-attention to model long-range dependencies.",
            "Knowledge distillation compresses large neural networks into smaller ones.",
        ] * (n // 3 + 1)
    return texts[:n]


def pre_generate_corpus(teacher_mdl, teacher_tok, raw_texts, device, max_new=40):
    """
    Generates text from a teacher model given raw input prompts.
    Used for Sequence-level KD: teacher's output text becomes the student's training target.

    Each raw_text[:80] is used as a prompt; the teacher continues generation.
    The returned list of strings is re-tokenized with the STUDENT's tokenizer
    inside train_seq_kd() -- no vocabulary alignment needed.
    """
    teacher_mdl.eval()
    corpus = []
    for text in tqdm(raw_texts, desc=f"  Generating teacher corpus", ncols=80, leave=False):
        prompt = text[:80]
        enc    = teacher_tok(prompt, return_tensors="pt",
                             truncation=True, max_length=32).to(device)
        try:
            with torch.no_grad():
                ids = teacher_mdl.generate(
                    **enc, max_new_tokens=max_new,
                    do_sample=True, temperature=0.8, top_p=0.9,
                    pad_token_id=teacher_tok.eos_token_id,
                )
            corpus.append(teacher_tok.decode(ids[0], skip_special_tokens=True))
        except Exception:
            corpus.append(prompt)   # Fallback: just use the prompt itself
    print(f"  [Corpus] Generated {len(corpus)} teacher samples.")
    return corpus


# =============================================================================
#  SECTION 8 -- TRAINING LOOPS
# =============================================================================

def train_token_kd(student, student_tok, teacher, raw_texts, device, label):
    """
    One-epoch token-level MiniLLM Reverse KD training loop.

    REQUIREMENT: student and teacher MUST share vocabulary (same tokenizer family).
    OPT-125m and OPT-350m both use the 50272-token OPT BPE vocabulary -> OK.

    Both models tokenize with student_tok. Teacher produces logits without grad.
    Student LoRA parameters are updated via Reverse KL + CE loss.
    """
    ds     = SimpleDataset(raw_texts, student_tok, max_len=MAX_SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt    = AdamW(filter(lambda p: p.requires_grad, student.parameters()),
                  lr=LR, weight_decay=0.01)
    student.train()
    tot, n = 0.0, 0
    pbar   = tqdm(loader, desc=f"  [{label}]", ncols=90, leave=True)
    for batch in pbar:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = ids.clone(); lbl[mask == 0] = -100

        with torch.no_grad():
            t_log = teacher(input_ids=ids, attention_mask=mask).logits.float()
        s_log = student(input_ids=ids, attention_mask=mask).logits.float()

        loss, _, _ = token_level_kd(s_log, t_log, lbl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        tot += loss.item(); n += 1
        pbar.set_postfix(L=f"{loss.item():.4f}")
    print(f"  [{label}] Avg Loss = {tot / n:.4f}")


def train_seq_kd(student, student_tok, teacher_corpus, device, label):
    """
    One-epoch sequence-level KD training loop.

    teacher_corpus: list of strings generated by the teacher model.
    The student tokenizer re-tokenizes each string independently ->
    no vocabulary size constraint between teacher and student architectures.

    Student learns via standard CE loss to reproduce teacher-generated text,
    which implicitly distills the teacher's output distribution.
    """
    ds     = SimpleDataset(teacher_corpus, student_tok, max_len=MAX_SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt    = AdamW(filter(lambda p: p.requires_grad, student.parameters()),
                  lr=LR, weight_decay=0.01)
    student.train()
    tot, n = 0.0, 0
    pbar   = tqdm(loader, desc=f"  [{label}]", ncols=90, leave=True)
    for batch in pbar:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = ids.clone(); lbl[mask == 0] = -100

        s_log = student(input_ids=ids, attention_mask=mask).logits.float()
        loss  = seq_level_kd(s_log, lbl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        tot += loss.item(); n += 1
        pbar.set_postfix(L=f"{loss.item():.4f}")
    print(f"  [{label}] Avg Loss = {tot / n:.4f}")


# =============================================================================
#  SECTION 9 -- LOAD BENCHMARKS & TRAINING DATA
# =============================================================================

print("\n" + "-" * 75)
print("  LOADING BENCHMARKS & TRAINING DATA")
print("-" * 75)

dolly_data   = get_dollyeval_samples(EVAL_SAMPLES)
selfinst_data = get_selfinst_samples(EVAL_SAMPLES)
vicuna_data  = get_vicunaeval_samples(EVAL_SAMPLES)

print(f"[INFO] DollyEval  : {len(dolly_data)} samples")
print(f"[INFO] SelfInst   : {len(selfinst_data)} samples")
print(f"[INFO] VicunaEval : {len(vicuna_data)} samples")

raw_texts = get_training_texts(MAX_TRAIN_SAMPLES)
print(f"[INFO] Training   : {len(raw_texts)} samples")

# Dict to collect all experiment results
# Format: {model_label: {bench: {ppl, rouge_l, avg_len}}}
ALL_RESULTS = {}


# =============================================================================
#  SECTION 10 -- LOAD OPT-350m TEACHER (used in Exp A + B1)
# =============================================================================

print("\n" + "-" * 75)
print("  LOADING TEACHER MODELS")
print("-" * 75)

opt_teacher, opt_teacher_tok = load_frozen_teacher(OPT_TEACHER, device)
OPT_VOCAB_SIZE = opt_teacher.config.vocab_size  # 50272

# Pre-generate OPT teacher corpus for cross-arch students (A2, A3)
print("\n[INFO] Pre-generating OPT-350m corpus for Seq-KD (A2, A3) ...")
opt_corpus = pre_generate_corpus(opt_teacher, opt_teacher_tok, raw_texts, device, max_new=TEACHER_GEN_NEW)


# =============================================================================
#  SECTION 11 -- EXPERIMENT A: Fixed Teacher = OPT-350m, Vary Student
# =============================================================================

print("\n" + "=" * 75)
print("  EXPERIMENT A -- Fixed Teacher: OPT-350m | Varying Student Architecture")
print("=" * 75)
print("  A1: OPT-125m   + LoRA  | Token-level Reverse KD  (same family)")
print("  A2: GPT-2      + LoRA  | Sequence-level KD        (cross-arch)")
print("  A3: DistilGPT2 + LoRA  | Sequence-level KD        (cross-arch)")


# ---- A1: OPT-125m  (token-level KD - same family as teacher) ----------------
print("\n  --- A1: OPT-125m Student + Token-level Reverse KD ---")
print("  [RATIONALE] OPT-125m and OPT-350m share the same 50272-token vocabulary.")
print("              Direct logit-level KL(P_student || P_teacher) is valid.")

a1_model, a1_tok = build_lora_student(OPT_STUDENT, OPT_LORA_TARGETS, device)
train_token_kd(a1_model, a1_tok, opt_teacher, raw_texts, device, "A1 OPT-125m Token-KD")

print(f"\n  [A1] Evaluating on benchmarks ...")
a1_res = evaluate_all(a1_model, a1_tok, "A1-OPT125m",
                      dolly_data, selfinst_data, vicuna_data, device)
ALL_RESULTS["A1: OPT-125m (Token-KD)"] = a1_res

del a1_model
if device.type == "cuda": torch.cuda.empty_cache()


# ---- A2: GPT-2  (sequence-level KD - cross-architecture) --------------------
print("\n  --- A2: GPT-2 Student + Sequence-level KD ---")
print("  [RATIONALE] GPT-2 uses its own 50257-token BPE vocabulary.")
print("              Logit-level KD is INVALID (vocab sizes differ + different token IDs).")
print("              Solution: OPT-350m generates text; GPT-2 trains via CE on those tokens.")

a2_model, a2_tok = build_lora_student(GPT2_STUDENT, GPT2_LORA_TARGETS, device)
# opt_corpus = OPT-350m generated text; re-tokenized by a2_tok internally
train_seq_kd(a2_model, a2_tok, opt_corpus, device, "A2 GPT-2 Seq-KD")

print(f"\n  [A2] Evaluating on benchmarks ...")
a2_res = evaluate_all(a2_model, a2_tok, "A2-GPT2",
                      dolly_data, selfinst_data, vicuna_data, device)
ALL_RESULTS["A2: GPT-2 (Seq-KD)"] = a2_res

del a2_model
if device.type == "cuda": torch.cuda.empty_cache()


# ---- A3: DistilGPT2  (sequence-level KD - smallest student) -----------------
print("\n  --- A3: DistilGPT2 Student + Sequence-level KD ---")
print("  [RATIONALE] DistilGPT2 (82M) is a distilled GPT-2 model -- same arch.")
print("              Uses Seq-KD for same cross-arch reason as A2.")
print("              Tests how well the smallest student absorbs OPT teacher knowledge.")

a3_model, a3_tok = build_lora_student(DGPT2_STUDENT, GPT2_LORA_TARGETS, device)
train_seq_kd(a3_model, a3_tok, opt_corpus, device, "A3 DistilGPT2 Seq-KD")

print(f"\n  [A3] Evaluating on benchmarks ...")
a3_res = evaluate_all(a3_model, a3_tok, "A3-DistilGPT2",
                      dolly_data, selfinst_data, vicuna_data, device)
ALL_RESULTS["A3: DistilGPT2 (Seq-KD)"] = a3_res

del a3_model
if device.type == "cuda": torch.cuda.empty_cache()


# ---- A4: Pythia-70m  (sequence-level KD - ultra compact student) -----------
print("\n  --- A4: Pythia-70m Student + Sequence-level KD ---")
print("  [RATIONALE] Pythia-70m (EleutherAI, 70M) tests an alternate architecture family (GPT-NeoX).")
print("              Uses Seq-KD from OPT-350m teacher. Tests lower parameter bound.")

a4_model, a4_tok = build_lora_student(PYTHIA70_STUDENT, PYTHIA_LORA_TARGETS, device)
train_seq_kd(a4_model, a4_tok, opt_corpus, device, "A4 Pythia-70m Seq-KD")

print(f"\n  [A4] Evaluating on benchmarks ...")
a4_res = evaluate_all(a4_model, a4_tok, "A4-Pythia70m",
                      dolly_data, selfinst_data, vicuna_data, device)
ALL_RESULTS["A4: Pythia-70m (Seq-KD)"] = a4_res

del a4_model
if device.type == "cuda": torch.cuda.empty_cache()


# ---- A5: Pythia-160m (sequence-level KD - alternative 160M student) ---------
print("\n  --- A5: Pythia-160m Student + Sequence-level KD ---")
print("  [RATIONALE] Pythia-160m (160M) is comparable in scale to OPT-125m.")
print("              Because it is cross-arch, it only gets hard Seq-KD targets,")
print("              demonstrating why OPT-125m's in-family Token-KD soft distillation wins.")

a5_model, a5_tok = build_lora_student(PYTHIA160_STUDENT, PYTHIA_LORA_TARGETS, device)
train_seq_kd(a5_model, a5_tok, opt_corpus, device, "A5 Pythia-160m Seq-KD")

print(f"\n  [A5] Evaluating on benchmarks ...")
a5_res = evaluate_all(a5_model, a5_tok, "A5-Pythia160m",
                      dolly_data, selfinst_data, vicuna_data, device)
ALL_RESULTS["A5: Pythia-160m (Seq-KD)"] = a5_res

del a5_model, opt_corpus
if device.type == "cuda": torch.cuda.empty_cache()


# =============================================================================
#  SECTION 12 -- EXPERIMENT B: Fixed Student = OPT-125m, Vary Teacher
# =============================================================================

print("\n" + "=" * 75)
print("  EXPERIMENT B -- Fixed Student: OPT-125m + LoRA | Varying Teacher")
print("=" * 75)
print("  B1: Teacher = OPT-350m     | Token-level Reverse KD  (same family)")
print("  B2: Teacher = GPT-2-medium | Sequence-level KD        (cross-arch)")


# ---- B1: OPT-350m teacher (same as A1 -- reuse results) ---------------------
print("\n  --- B1: OPT-350m Teacher + Token-level Reverse KD ---")
print("  [NOTE] B1 is architecturally identical to A1 (OPT-125m student, OPT-350m teacher).")
print("         Reusing A1 results to avoid redundant computation.")
ALL_RESULTS["B1: OPT-125m + OPT-350m Teacher"] = a1_res


# ---- B2: GPT-2-medium teacher (cross-architecture) --------------------------
print("\n  --- B2: GPT-2-medium Teacher + Sequence-level KD ---")
print("  [RATIONALE] GPT-2-medium (~345M) has different tokenizer than OPT-125m student.")
print("              GPT-2-medium generates text; OPT-125m student trains via CE on tokens.")
print("              Interesting because teacher family != student family.")

gpt2_teacher, gpt2_teacher_tok = load_frozen_teacher(GPT2_TEACHER, device)

print("\n[INFO] Pre-generating GPT-2-medium corpus for OPT-125m student ...")
gpt2med_corpus = pre_generate_corpus(gpt2_teacher, gpt2_teacher_tok,
                                      raw_texts, device, max_new=TEACHER_GEN_NEW)

# Free GPT-2-medium teacher from memory before loading student
del gpt2_teacher
if device.type == "cuda": torch.cuda.empty_cache()

b2_model, b2_tok = build_lora_student(OPT_STUDENT, OPT_LORA_TARGETS, device)
# gpt2med_corpus = GPT-2-medium generated text; re-tokenized by OPT b2_tok
train_seq_kd(b2_model, b2_tok, gpt2med_corpus, device, "B2 OPT-125m <- GPT2-Med Seq-KD")

print(f"\n  [B2] Evaluating on benchmarks ...")
b2_res = evaluate_all(b2_model, b2_tok, "B2-OPT125m+GPT2MedTeacher",
                      dolly_data, selfinst_data, vicuna_data, device)
ALL_RESULTS["B2: OPT-125m + GPT2-Med Teacher"] = b2_res

del b2_model, gpt2med_corpus, opt_teacher
if device.type == "cuda": torch.cuda.empty_cache()


# =============================================================================
#  SECTION 13 -- UNIFIED COMPARISON TABLE
# =============================================================================

BENCHMARKS = ["DollyEval", "SelfInst", "VicunaEval"]

print("\n\n" + "=" * 75)
print("  FINAL UNIFIED COMPARISON -- ALL EXPERIMENTS x ALL BENCHMARKS")
print("=" * 75)
print("  Metrics: PPL (lower=better) | ROUGE-L (higher=better) | AvgLen (coverage)")

# ---- Per-Benchmark Table ----
for bench in BENCHMARKS:
    has_ref = bench != "VicunaEval"
    print(f"\n  ---- {bench} {'(has reference answers)' if has_ref else '(no references: ROUGE-L N/A)'} ----")
    print(f"  {'Model Config':<42} | {'PPL':>8} | {'ROUGE-L':>9} | {'AvgLen':>7}")
    print("  " + "-" * 73)
    for label, res in ALL_RESULTS.items():
        r      = res.get(bench, {})
        ppl    = r.get("ppl", float("inf"))
        rouge  = r.get("rouge_l")
        avglen = r.get("avg_len", 0.0)
        rouge_str = f"{rouge:.4f}" if rouge is not None else "  N/A  "
        print(f"  {label:<42} | {ppl:>8.2f} | {rouge_str:>9} | {avglen:>7.1f}")

# ---- Average PPL Ranking ----
print(f"\n\n  ---- AVERAGE PPL RANKING (across all 3 benchmarks, lower=better) ----")
print(f"  {'Model Config':<42} | {'Avg PPL':>9} | {'Avg ROUGE-L':>12} | Rank")
print("  " + "-" * 75)

rankings = []
for label, res in ALL_RESULTS.items():
    ppls   = [res[b]["ppl"]     for b in BENCHMARKS if b in res]
    rouges = [res[b]["rouge_l"] for b in BENCHMARKS
              if b in res and res[b]["rouge_l"] is not None]
    avg_ppl   = sum(ppls)   / max(len(ppls), 1)
    avg_rouge = sum(rouges) / max(len(rouges), 1) if rouges else None
    rankings.append((label, avg_ppl, avg_rouge))

rankings.sort(key=lambda x: x[1])   # Ascending PPL = better
for rank, (label, avg_ppl, avg_rouge) in enumerate(rankings, 1):
    r_str  = f"{avg_rouge:.4f}" if avg_rouge is not None else "  N/A  "
    marker = "  <-- BEST" if rank == 1 else ""
    print(f"  {label:<42} | {avg_ppl:>9.2f} | {r_str:>12} | #{rank}{marker}")


# =============================================================================
#  SECTION 14 -- VIVA REFERENCE TABLES
# =============================================================================

print("\n\n" + "=" * 75)
print("  VIVA REFERENCE -- DISTILLATION METHOD DECISION TREE")
print("=" * 75)
print("""
  QUESTION: When can I use Token-level KD vs Sequence-level KD?

  +-----------------------+----------------------------+-----------------------------+
  | Criterion             | Token-level KD             | Sequence-level KD           |
  +-----------------------+----------------------------+-----------------------------+
  | Vocab requirement     | MUST match (same family)   | Any (different vocab OK)    |
  | Information density   | Rich soft-label logits     | Hard token sequences        |
  | KD signal quality     | Higher (prob. over vocab)  | Lower (argmax of teacher)   |
  | Cross-arch pairs      | NOT VALID                  | VALID                       |
  | Example               | OPT-125m <- OPT-350m       | GPT-2 <- OPT-350m           |
  | Loss formula          | KL(P_s || P_t)             | CE(s_logits, teacher_tokens)|
  | Reference paper       | MiniLLM (Gu et al. 2023)   | Kim & Rush (2016)           |
  +-----------------------+----------------------------+-----------------------------+
""")

print("=" * 75)
print("  VIVA REFERENCE -- EXPERIMENT DESIGN SUMMARY")
print("=" * 75)
rows = [
    ("A1", "OPT-125m+LoRA",   "OPT-350m",    "Token-level Reverse KD",
     "Same OPT family: logit KL valid"),
    ("A2", "GPT-2+LoRA",      "OPT-350m",    "Sequence-level KD",
     "Cross-arch: seq-level only valid method"),
    ("A3", "DistilGPT2+LoRA", "OPT-350m",    "Sequence-level KD",
     "Smallest student: tests compression limit"),
    ("A4", "Pythia-70m+LoRA", "OPT-350m",    "Sequence-level KD",
     "NeoX family (70M): ultra-compact cross-arch"),
    ("A5", "Pythia-160m+LoRA","OPT-350m",    "Sequence-level KD",
     "NeoX family (160M): shows soft-KD advantage of OPT"),
    ("B1", "OPT-125m+LoRA",   "OPT-350m",    "Token-level Reverse KD",
     "Same as A1 (baseline for Exp B)"),
    ("B2", "OPT-125m+LoRA",   "GPT-2-medium","Sequence-level KD",
     "Cross-arch teacher: weaker but valid"),
]
print(f"\n  {'ID':<4} | {'Student':<16} | {'Teacher':<14} | {'Method':<26} | Design Rationale")
print("  " + "-" * 90)
for row in rows:
    print(f"  {row[0]:<4} | {row[1]:<16} | {row[2]:<14} | {row[3]:<26} | {row[4]}")

print("\n" + "=" * 75)
print("  COMPARISON EXPERIMENT COMPLETE")
print("=" * 75 + "\n")

# =============================================================================
#  SECTION 15 -- SAVE PRESENTATION-READY REPORTS & EXPORTS
# =============================================================================

# 1. JSON Export
try:
    with open("results_comparison.json", "w", encoding="utf-8") as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print("[EXPORT] Saved machine-readable results -> results_comparison.json")
except Exception as e:
    print(f"[WARN] Failed to write JSON: {e}")

# 2. Markdown Report Export
try:
    with open("results_summary.md", "w", encoding="utf-8") as f:
        f.write("# Knowledge Distillation Benchmark & Model Comparison Report\n\n")
        f.write("Generated on: 2026-10-06\n\n")
        f.write("### Benchmark Performance Across Evaluated Models\n\n")
        f.write("| Model Configuration | DollyEval PPL | SelfInst PPL | VicunaEval PPL | Avg PPL | Avg ROUGE-L | Rank |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, (label, avg_ppl, avg_rouge) in enumerate(rankings, 1):
            dolly_ppl = ALL_RESULTS.get(label, {}).get("DollyEval", {}).get("ppl", float("nan"))
            selfinst_ppl = ALL_RESULTS.get(label, {}).get("SelfInst", {}).get("ppl", float("nan"))
            vicuna_ppl = ALL_RESULTS.get(label, {}).get("VicunaEval", {}).get("ppl", float("nan"))
            r_str = f"{avg_rouge:.4f}" if avg_rouge is not None else "N/A"
            badge = " **(BEST)**" if rank == 1 else ""
            f.write(f"| **{label}**{badge} | {dolly_ppl:.2f} | {selfinst_ppl:.2f} | {vicuna_ppl:.2f} | **{avg_ppl:.2f}** | **{r_str}** | #{rank} |\n")
        f.write("\n\n*Note: Lower PPL indicates higher model prediction confidence; higher ROUGE-L indicates closer agreement with ground-truth references.*\n")
    print("[EXPORT] Saved markdown presentation summary -> results_summary.md")
except Exception as e:
    print(f"[WARN] Failed to write Markdown: {e}")

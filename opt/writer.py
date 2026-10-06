# =============================================================================
#  MiniLLM-Based Knowledge Distillation  .  main.py
#  Author  : Afsal
#  Date    : 2026-10-02
#
#  PURPOSE
#  -------
#  This single script implements the full MiniLLM knowledge-distillation
#  pipeline.  Knowledge Distillation (KD) trains a small student model to
#  mimic a larger, more capable teacher model.  Instead of training the
#  student only on ground-truth labels, we also teach it to match the
#  teacher output probability distribution, capturing dark knowledge
#  (soft information) that plain labels do not carry.
#
#  MiniLLM NOVELTY
#  ---------------
#  Classic KD uses Forward-KL: KL(P_teacher || P_student) which averages
#  over all tokens the teacher assigns probability to, including many low-
#  probability ones the student will never generate.  MiniLLM instead
#  minimises Reverse-KL: KL(P_student || P_teacher) which forces the
#  student to be mode-seeking, concentrating on regions of high teacher
#  probability and avoiding wasting capacity on unlikely tokens.
#
#  EXECUTION ORDER
#  Section 1 : Setup and device check
#  Section 2 : Load tokenizer, teacher model, student model + LoRA
#  Section 3 : Define minillm_reverse_kd_loss()
#  Section 4 : Baseline distillation loop on WikiText-2
#  Section 5 : Medical domain distillation loop on PubMedQA
#  Section 6 : Save LoRA adapter weights and ZIP the output folder
#  Section 7 : Medical text generation test (inference verification)
# =============================================================================
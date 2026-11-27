#!/usr/bin/env python3
"""
Paired sentence-level bootstrap significance testing for Nepali NER evaluations.

Implements non-parametric paired bootstrap resampling (Koehn, 2004;
Berg-Kirkpatrick et al., 2012) across the N=2,078 held-out test sentences
to compute empirical p-values and 95% confidence intervals for Delta F1
across 0-shot, 5-shot, and 10-shot generative LLM configurations.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

EVAL_FILES: Dict[str, str] = {
    "Llama 3.3 70B (10-shot)": "results/llm_api/meta-llama_llama-3.3-70b-instruct_shots10_n2078_seed42.jsonl",
    "Llama 3.3 70B (5-shot)": "results/llm_api/meta-llama_llama-3.3-70b-instruct_shots5_n2078_seed42.jsonl",
    "Llama 3.3 70B (0-shot)": "results/llm_api/meta-llama_llama-3.3-70b-instruct_shots0_n2078_seed42.jsonl",
    "Qwen 2.5 72B (10-shot)": "results/llm_api/qwen_qwen-2.5-72b-instruct_shots10_n2078_seed42.jsonl",
    "Qwen 2.5 72B (5-shot)": "results/llm_api/qwen_qwen-2.5-72b-instruct_shots5_n2078_seed42.jsonl",
    "Qwen 2.5 72B (0-shot)": "results/llm_api/qwen_qwen-2.5-72b-instruct_shots0_n2078_seed42.jsonl",
    "GPT-4o-mini (10-shot)": "results/llm_api/openai_gpt-4o-mini_shots10_n2078_seed42.jsonl",
    "GPT-4o-mini (5-shot)": "results/llm_api/openai_gpt-4o-mini_shots5_n2078_seed42.jsonl",
    "GPT-4o-mini (0-shot)": "results/llm_api/openai_gpt-4o-mini_shots0_n2078_seed42.jsonl",
    "Llama 3.1 8B (10-shot)": "results/llm_api/meta-llama_llama-3.1-8b-instruct_shots10_n2078_seed42.jsonl",
    "Llama 3.1 8B (5-shot)": "results/llm_api/meta-llama_llama-3.1-8b-instruct_shots5_n2078_seed42.jsonl",
    "Llama 3.1 8B (0-shot)": "results/llm_api/meta-llama_llama-3.1-8b-instruct_shots0_n2078_seed42.jsonl",
}

COMPARISONS: List[Tuple[str, str]] = [
    # Intra-family shot scaling: 5-shot vs 0-shot
    ("Llama 3.3 70B (5-shot)", "Llama 3.3 70B (0-shot)"),
    ("Qwen 2.5 72B (5-shot)", "Qwen 2.5 72B (0-shot)"),
    ("GPT-4o-mini (5-shot)", "GPT-4o-mini (0-shot)"),
    ("Llama 3.1 8B (5-shot)", "Llama 3.1 8B (0-shot)"),
    # Intra-family shot scaling: 10-shot vs 0-shot
    ("Llama 3.3 70B (10-shot)", "Llama 3.3 70B (0-shot)"),
    ("Qwen 2.5 72B (10-shot)", "Qwen 2.5 72B (0-shot)"),
    ("GPT-4o-mini (10-shot)", "GPT-4o-mini (0-shot)"),
    ("Llama 3.1 8B (10-shot)", "Llama 3.1 8B (0-shot)"),
    # Intra-family shot scaling: 10-shot vs 5-shot
    ("Llama 3.3 70B (10-shot)", "Llama 3.3 70B (5-shot)"),
    ("Qwen 2.5 72B (10-shot)", "Qwen 2.5 72B (5-shot)"),
    ("GPT-4o-mini (10-shot)", "GPT-4o-mini (5-shot)"),
    ("Llama 3.1 8B (10-shot)", "Llama 3.1 8B (5-shot)"),
    # Cross-family comparisons (5-shot)
    ("Llama 3.3 70B (5-shot)", "Qwen 2.5 72B (5-shot)"),
    ("Llama 3.3 70B (5-shot)", "GPT-4o-mini (5-shot)"),
    ("Qwen 2.5 72B (5-shot)", "GPT-4o-mini (5-shot)"),
    ("Llama 3.3 70B (5-shot)", "Llama 3.1 8B (5-shot)"),
    # Cross-family comparisons (10-shot & peak configurations)
    ("Llama 3.3 70B (10-shot)", "GPT-4o-mini (10-shot)"),
    ("Llama 3.3 70B (10-shot)", "Qwen 2.5 72B (10-shot)"),
    ("GPT-4o-mini (10-shot)", "Qwen 2.5 72B (10-shot)"),
    ("GPT-4o-mini (10-shot)", "Qwen 2.5 72B (5-shot)"),
    ("Llama 3.3 70B (5-shot)", "GPT-4o-mini (10-shot)"),
]


def load_sentence_counts(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-sentence strict span TP, FP, and FN counts."""
    tp_list, fp_list, fn_list = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "__summary__" in record:
                continue
            gold_ents = set(get_entities(record["gold"]))
            pred_ents = set(get_entities(record["pred"]))
            tp_list.append(len(gold_ents & pred_ents))
            fp_list.append(len(pred_ents - gold_ents))
            fn_list.append(len(gold_ents - pred_ents))
    return np.array(tp_list), np.array(fp_list), np.array(fn_list)


def main(n_iterations: int = 10000, seed: int = 42) -> None:
    """Execute paired bootstrap significance tests across model pairs."""
    counts = {
        name: load_sentence_counts(Path(path))
        for name, path in EVAL_FILES.items()
    }

    n_sentences = len(next(iter(counts.values()))[0])
    np.random.seed(seed)
    sample_indices = np.random.randint(0, n_sentences, size=(n_iterations, n_sentences))

    boot_f1: Dict[str, np.ndarray] = {}
    point_f1: Dict[str, float] = {}
    for name, (tp, fp, fn) in counts.items():
        total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
        point_f1[name] = float(2 * total_tp / (2 * total_tp + total_fp + total_fn) * 100)
        b_tp = tp[sample_indices].sum(axis=1)
        b_fp = fp[sample_indices].sum(axis=1)
        b_fn = fn[sample_indices].sum(axis=1)
        boot_f1[name] = 2 * b_tp / (2 * b_tp + b_fp + b_fn) * 100

    results = []
    for model_a, model_b in COMPARISONS:
        diff = boot_f1[model_a] - boot_f1[model_b]
        point_diff = point_f1[model_a] - point_f1[model_b]
        ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
        p_one_sided = float(np.mean(diff <= 0)) if point_diff >= 0 else float(np.mean(diff >= 0))
        p_two_sided = float(2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))

        results.append({
            "model_a": model_a,
            "model_b": model_b,
            "f1_a_pct": round(point_f1[model_a], 2),
            "f1_b_pct": round(point_f1[model_b], 2),
            "delta_f1_pct": round(point_diff, 2),
            "paired_95_ci_pct": [round(float(ci_low), 2), round(float(ci_high), 2)],
            "p_value_one_sided": round(p_one_sided, 6),
            "p_value_two_sided": round(p_two_sided, 6),
            "n_iterations": n_iterations,
            "n_sentences": n_sentences,
            "seed": seed,
        })

    out_path = Path("results/comparison/paired_significance_tests.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved paired bootstrap significance results to {out_path}")
    for r in results:
        print(
            f"{r['model_a']:25} vs {r['model_b']:25} | "
            f"dF1={r['delta_f1_pct']:+6.2f}% | CI={r['paired_95_ci_pct']} | p(2-sided)={r['p_value_two_sided']:.4f}"
        )


if __name__ == "__main__":
    main()

"""
Script to generate all publication-ready figures for the EverestNER paper:
1. barchart.png: F1-score comparison with 95% Bootstrap Confidence Intervals
2. heatmap2.png: Per-entity F1-score heatmap
3. trainingtime.png: Latency vs F1 trade-off scatter plot
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#E0E0E0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# ==========================================
# 1. BAR CHART: Overall F1 with Error Bars
# ==========================================
def plot_barchart():
    models = [
        "XLM-RoBERTa\n(Fine-tuned)",
        "mBERT\n(Fine-tuned)",
        "DistilBERT\n(Fine-tuned)",
        "Llama 3.3 70B\n(5-shot)",
        "Llama 3.3 70B\n(0-shot)",
        "Qwen 2.5 72B\n(5-shot)",
        "Qwen 2.5 72B\n(0-shot)",
        "GPT-4o-mini\n(5-shot)",
        "GPT-4o-mini\n(0-shot)",
        "Llama 3.1 8B\n(5-shot)",
        "Llama 3.1 8B\n(0-shot)",
    ]

    f1_scores = [
        90.60, 87.46, 82.94,
        70.78, 67.53,
        63.26, 57.38,
        58.58, 53.85,
        52.97, 35.98
    ]

    margins = [
        1.04, 1.23, 1.47,
        1.77, 1.88,
        1.86, 1.76,
        2.27, 2.20,
        2.14, 1.66
    ]

    colors = [
        "#1f77b4", "#2ca02c", "#17becf",
        "#ff7f0e", "#fdae61",
        "#9467bd", "#c5b0d5",
        "#d62728", "#ff9896",
        "#8c564b", "#c49c94"
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(models))

    bars = ax.bar(x, f1_scores, yerr=margins, capsize=4, color=colors, edgecolor='#333333', linewidth=0.8, alpha=0.9)

    ax.set_ylabel("Strict Entity F1-Score (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.7)

    # Annotate values above bars
    for bar, score in zip(bars, f1_scores):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 3.0, f"{score:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Draw divider line between encoders and LLMs
    ax.axvline(x=2.5, color='gray', linestyle=':', linewidth=1.5)
    ax.text(1.0, 93, "Discriminative Encoders", ha='center', fontsize=11, fontweight='bold', color='#1f77b4', bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef7ff', edgecolor='#1f77b4', alpha=0.8))
    ax.text(6.5, 93, "Generative LLM Prompting", ha='center', fontsize=11, fontweight='bold', color='#ff7f0e', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5eb', edgecolor='#ff7f0e', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "barchart.png", dpi=300)
    print("Saved assets/barchart.png")
    plt.close()


# ==========================================
# 2. HEATMAP: Per-Entity Breakdown
# ==========================================
def plot_heatmap():
    model_labels = [
        "XLM-RoBERTa (Fine-tuned)",
        "mBERT (Fine-tuned)",
        "DistilBERT (Fine-tuned)",
        "Llama 3.3 70B (5-shot)",
        "Llama 3.3 70B (0-shot)",
        "Qwen 2.5 72B (5-shot)",
        "Qwen 2.5 72B (0-shot)",
        "GPT-4o-mini (5-shot)",
        "GPT-4o-mini (0-shot)",
        "Llama 3.1 8B (5-shot)",
        "Llama 3.1 8B (0-shot)",
    ]

    entities = [
        "PERSON\n(N=1173)",
        "LOCATION\n(N=736)",
        "ORGANIZATION\n(N=671)",
        "DATE\n(N=538)",
        "EVENT\n(N=49)",
    ]

    data = np.array([
        [95.6, 88.9, 88.1, 89.2, 53.2],  # XLM-R
        [92.2, 86.3, 83.4, 87.3, 55.2],  # mBERT
        [87.6, 82.0, 76.3, 86.6, 43.3],  # DistilBERT
        [84.1, 75.5, 61.6, 50.8, 32.5],  # Llama 3.3 5s
        [81.8, 71.0, 55.7, 45.6, 35.4],  # Llama 3.3 0s
        [73.3, 69.0, 55.9, 53.1, 16.0],  # Qwen 2.5 5s
        [70.4, 65.3, 47.2, 49.8,  9.9],  # Qwen 2.5 0s
        [73.0, 52.5, 51.7, 47.6, 28.2],  # GPT-4o-mini 5s
        [70.7, 52.6, 35.7, 39.2, 26.4],  # GPT-4o-mini 0s
        [69.2, 60.3, 38.8, 25.8,  4.7],  # Llama 3.1 8B 5s
        [45.1, 46.3, 31.5, 37.0,  2.5],  # Llama 3.1 8B 0s
    ])

    fig, ax = plt.subplots(figsize=(8, 7.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=entities,
        yticklabels=model_labels,
        cbar_kws={'label': 'Entity F1-Score (%)'},
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=100
    )

    ax.set_title("Entity-Specific F1-Scores across Models on EverestNER", pad=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap2.png", dpi=300)
    print("Saved assets/heatmap2.png")
    plt.close()


# ==========================================
# 3. SCATTER PLOT: Latency vs F1-Score
# ==========================================
def plot_trainingtime():
    # Encoders: Inference latency in seconds (~0.02s) vs F1
    enc_models = ["XLM-RoBERTa", "mBERT", "DistilBERT"]
    enc_latency = [0.025, 0.020, 0.012]
    enc_f1 = [90.60, 87.46, 82.94]

    # LLMs: Inference latency in seconds vs F1
    llm_models = [
        "Llama 3.3 70B (5s)", "Llama 3.3 70B (0s)",
        "Qwen 2.5 72B (5s)", "Qwen 2.5 72B (0s)",
        "GPT-4o-mini (5s)", "GPT-4o-mini (0s)",
        "Llama 3.1 8B (5s)", "Llama 3.1 8B (0s)"
    ]
    llm_latency = [1.17, 1.01, 1.88, 2.22, 1.18, 1.06, 5.45, 3.18]
    llm_f1 = [70.78, 67.53, 63.26, 57.38, 58.58, 53.85, 52.97, 35.98]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Plot Encoders
    ax.scatter(enc_latency, enc_f1, color='#1f77b4', s=160, marker='o', label='Fine-tuned Encoders (Local GPU)', edgecolors='black', zorder=5)
    for name, x, y in zip(enc_models, enc_latency, enc_f1):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, -4), fontweight='bold', color='#1f77b4')

    # Plot LLMs
    ax.scatter(llm_latency, llm_f1, color='#ff7f0e', s=160, marker='^', label='Generative LLMs (API Prompting)', edgecolors='black', zorder=5)
    for name, x, y in zip(llm_models, llm_latency, llm_f1):
        offset = (-10, 10) if "8B (5s)" in name else (8, -4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset, fontsize=9, color='#333333')

    ax.set_xscale('log')
    ax.set_xlabel("Average Inference Latency per Sentence (seconds, log-scale)")
    ax.set_ylabel("Strict Entity F1-Score (%)")
    ax.set_title("Inference Latency vs. F1-Score Trade-off", fontweight='bold', pad=12)
    ax.grid(True, which="both", alpha=0.6)
    ax.set_ylim(25, 98)
    ax.legend(loc='lower left', frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trainingtime.png", dpi=300)
    print("Saved assets/trainingtime.png")
    plt.close()


if __name__ == "__main__":
    plot_barchart()
    plot_heatmap()
    plot_trainingtime()
    print("All 3 publication figures generated successfully!")

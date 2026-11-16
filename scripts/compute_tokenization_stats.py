#!/usr/bin/env python3
"""
Compute tokenization fertility and subword fragmentation metrics
across evaluated tokenizers on the EverestNER held-out test split.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from transformers import AutoTokenizer
import tiktoken
import json

def load_sentences(test_file: str):
    sentences = []
    curr = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr:
                    sentences.append(curr)
                    curr = []
                continue
            parts = line.split()
            curr.append(parts[0])
        if curr:
            sentences.append(curr)
    return sentences

def main():
    test_file = "data/raw/EverestNER-test-bio.txt"
    sentences = load_sentences(test_file)
    print(f"Loaded {len(sentences)} sentences.")
    
    total_words = sum(len(s) for s in sentences)
    full_texts = [" ".join(s) for s in sentences]
    total_chars = sum(len(t) for t in full_texts)
    
    print(f"Total Words (whitespace-separated): {total_words}")
    print(f"Total Characters: {total_chars}")
    
    # Load tokenizers
    print("\nInitializing Tokenizers...")
    tokenizers = {
        "XLM-RoBERTa (SentencePiece)": {
            "type": "Encoder",
            "vocab": 250002,
            "fn": AutoTokenizer.from_pretrained("xlm-roberta-base")
        },
        "mBERT / DistilBERT (WordPiece)": {
            "type": "Encoder",
            "vocab": 119547,
            "fn": AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        },
        "GPT-4o-mini (o200k_base BPE)": {
            "type": "Generative LLM",
            "vocab": 200019,
            "fn": tiktoken.get_encoding("o200k_base")
        },
        "Qwen 2.5 (BPE)": {
            "type": "Generative LLM",
            "vocab": 151665,
            "fn": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        },
        "Llama 3 / 3.1 / 3.3 (BPE)": {
            "type": "Generative LLM",
            "vocab": 128256,
            "fn": AutoTokenizer.from_pretrained("bofenghuang/Meta-Llama-3-8B")
        }
    }
    
    results = []
    base_tokens = None
    
    for name, meta in tokenizers.items():
        print(f"Evaluating {name}...")
        tok = meta["fn"]
        total_subwords = 0
        
        for text in full_texts:
            if isinstance(tok, tiktoken.Encoding):
                subwords = len(tok.encode(text))
            else:
                # Disable special tokens for pure text measurement
                subwords = len(tok.tokenize(text))
            total_subwords += subwords
            
        fertility = total_subwords / total_words
        chars_per_token = total_chars / total_subwords
        
        if base_tokens is None:
            base_tokens = total_subwords
            expansion = 1.0
        else:
            expansion = total_subwords / base_tokens
            
        results.append({
            "Tokenizer": name,
            "Paradigm": meta["type"],
            "Vocab Size": f"{meta['vocab']:,}",
            "Total Subwords": total_subwords,
            "Fertility (Tokens/Word)": round(fertility, 3),
            "Chars/Token": round(chars_per_token, 2),
            "Expansion vs XLM-R": f"{expansion:.2f}x"
        })
        
    print("\n--- RESULTS ---")
    print(f"{'Tokenizer':<30} | {'Paradigm':<15} | {'Vocab':<10} | {'Subwords':<10} | {'Fertility':<10} | {'Chars/Tok':<10} | {'Expansion':<10}")
    print("-" * 110)
    for r in results:
        print(f"{r['Tokenizer']:<30} | {r['Paradigm']:<15} | {r['Vocab Size']:<10} | {r['Total Subwords']:<10} | {r['Fertility (Tokens/Word)']:<10.3f} | {r['Chars/Token']:<10.2f} | {r['Expansion vs XLM-R']:<10}")
        
    # Save to JSON and Markdown
    out_json = Path("results/comparison/tokenization_stats.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Generate LaTeX table snippet
    latex_table = """\\begin{table}[t]
\\centering
\\caption{Devanagari Tokenization Fertility and Fragmentation on EverestNER Test Partition ($N=2{,}078$ sentences, 40,019 words). Higher fertility indicates greater subword fragmentation.}
\\label{tab:tokenization}
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Tokenizer} & \\textbf{Type} & \\textbf{Vocab} & \\textbf{Subwords} & \\textbf{Fertility} & \\textbf{Chars/Tok} & \\textbf{Expansion} \\\\
\\midrule
"""
    for r in results:
        name_short = r['Tokenizer'].split(' (')[0]
        latex_table += f"\\textbf{{{name_short}}} & {r['Paradigm']} & {r['Vocab Size']} & {r['Total Subwords']:,} & {r['Fertility (Tokens/Word)']:.2f} & {r['Chars/Token']:.2f} & {r['Expansion vs XLM-R']} \\\\\n"
    latex_table += """\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    with open("results/comparison/tokenization_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print("\nSaved results to results/comparison/tokenization_stats.json and tokenization_table.tex")

if __name__ == "__main__":
    main()

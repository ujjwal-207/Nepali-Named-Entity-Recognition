| Model | Architecture | Shots | Precision (%) | Recall (%) | Strict F1 (%) | 95% Bootstrap CI | Latency | Test Cost ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XLM-RoBERTa Base** | Fine-Tuned (270M) | — | 88.47 | 92.83 | **90.60** $\pm$ 1.04 | [89.51, 91.59] | ~25 ms | $0.000 |
| **mBERT (cased)** | Fine-Tuned (178M) | — | 85.49 | 89.53 | **87.46** $\pm$ 1.23 | [86.18, 88.65] | ~20 ms | $0.000 |
| **DistilBERT Multilingual** | Fine-Tuned (135M) | — | 80.53 | 85.51 | **82.94** $\pm$ 1.47 | [81.42, 84.35] | ~12 ms | $0.000 |
| **Llama 3.3 70B Instruct** | Generative LLM (70B) | 5 | 69.41 | 72.21 | **70.78** $\pm$ 1.77 | [69.05, 72.60] | 1170 ms | $0.578 |
| **Llama 3.3 70B Instruct** | Generative LLM (70B) | 0 | 66.63 | 68.46 | **67.53** $\pm$ 1.88 | [65.41, 69.18] | 1010 ms | $0.148 |
| **Qwen 2.5 72B Instruct** | Generative LLM (72B) | 5 | 59.31 | 67.76 | **63.26** $\pm$ 1.86 | [61.34, 65.06] | 1880 ms | $0.727 |
| **Qwen 2.5 72B Instruct** | Generative LLM (72B) | 0 | 50.59 | 66.28 | **57.38** $\pm$ 1.76 | [55.56, 59.09] | 2220 ms | $0.179 |
| **GPT-4o-mini** | Proprietary | 5 | 74.63 | 48.22 | **58.58** $\pm$ 2.27 | [56.22, 60.76] | 1180 ms | $0.175 |
| **GPT-4o-mini** | Proprietary | 0 | 72.09 | 42.97 | **53.85** $\pm$ 2.20 | [51.53, 55.93] | 1060 ms | $0.058 |
| **Llama 3.1 8B Instruct** | Generative LLM (8B) | 5 | 53.44 | 52.51 | **52.97** $\pm$ 2.14 | [50.84, 55.12] | 5450 ms | $0.080 |
| **Llama 3.1 8B Instruct** | Generative LLM (8B) | 0 | 25.74 | 59.77 | **35.98** $\pm$ 1.66 | [34.37, 37.69] | 3180 ms | $0.025 |

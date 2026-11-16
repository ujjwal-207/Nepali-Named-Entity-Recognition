"""
Nepali Named Entity Recognition (NER) Benchmark.

Discriminative Fine-Tuning vs. Generative In-Context Prompting
on the EverestNER Corpus.
"""

from .baseline_trainer import NERDataset, NERTrainer
from .reproducibility import get_environment_info, reproducibility_block, set_seed

__version__ = "1.0.0"

__all__ = [
    "NERDataset",
    "NERTrainer",
    "set_seed",
    "reproducibility_block",
    "get_environment_info",
]

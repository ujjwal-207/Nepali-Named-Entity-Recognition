#!/usr/bin/env python3
"""
Evaluation script for trained discriminative transformer NER models.

Loads saved checkpoints, generates span-level seqeval metrics on the test split,
and calculates 95% bootstrap confidence intervals.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

sys.path.append(str(Path(__file__).parent.parent))
from src.baseline_trainer import NERTrainer
from src.reproducibility import reproducibility_block

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, str] = {
    "mbert": "bert-base-multilingual-cased",
    "distilbert": "distilbert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
}


def evaluate_model(
    model_dir: Path, model_key: str, batch_size: int = 64
) -> Optional[Dict[str, Any]]:
    """Evaluate a saved model checkpoint on the test split.

    Args:
        model_dir: Path to directory containing checkpoint.
        model_key: Identifier key in MODEL_REGISTRY.
        batch_size: Evaluation batch size.

    Returns:
        Dictionary of evaluation results or None if failed.
    """
    final_model_dir = model_dir / "final_model"
    if not final_model_dir.exists():
        logger.warning("Checkpoint not found at %s. Skipping.", final_model_dir)
        return None

    model_name = MODEL_REGISTRY.get(model_key)
    if not model_name:
        logger.error("Unknown model identifier: %s", model_key)
        return None

    try:
        trainer_helper = NERTrainer(
            model_name=model_name,
            data_dir=Path("data/raw"),
            output_dir=model_dir,
            max_length=64,
            batch_size=batch_size,
            seed=42,
        )
        trainer_helper.load_data()

        if trainer_helper.test_dataset is None:
            logger.error("Test dataset split not found.")
            return None

        model = AutoModelForTokenClassification.from_pretrained(str(final_model_dir))
        model.to(trainer_helper.device)

        eval_trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(model_dir),
                per_device_eval_batch_size=batch_size,
                report_to="none",
            ),
            compute_metrics=trainer_helper.compute_metrics,
        )

        results = trainer_helper.evaluate(eval_trainer)
        return results

    except Exception as e:
        logger.exception("Failed evaluation for %s: %s", model_key, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved transformer NER models.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/baseline",
        help="Root directory containing saved model subdirectories.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size per device (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    args = parser.parse_args()

    reproducibility_block(
        seed=args.seed,
        output_dir="results/evaluation",
        experiment_name="evaluate-trained-models",
    )

    base_dir = Path(args.models_dir)
    results_summary: Dict[str, Any] = {}

    for key in MODEL_REGISTRY:
        model_dir = base_dir / key
        if (model_dir / "final_model").exists():
            logger.info("Evaluating checkpoint: %s (batch_size=%d)", key, args.batch_size)
            res = evaluate_model(model_dir, key, batch_size=args.batch_size)
            if res:
                results_summary[key] = res
        else:
            logger.info("No checkpoint found for %s", key)

    if results_summary:
        print("\n" + "=" * 65)
        print(f"{'Model':<15} {'F1':<10} {'Precision':<12} {'Recall':<10} {'95% CI':<15}")
        print("-" * 65)
        for key, res in results_summary.items():
            f1 = res.get("f1", 0.0)
            p = res.get("precision", 0.0)
            r = res.get("recall", 0.0)
            low = res.get("f1_ci_low", 0.0)
            high = res.get("f1_ci_high", 0.0)
            print(f"{key:<15} {f1:<10.4f} {p:<12.4f} {r:<10.4f} [{low:.4f}, {high:.4f}]")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

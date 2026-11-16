#!/usr/bin/env python3
"""
Sequential training pipeline for baseline discriminative NER models.

Trains mBERT, XLM-RoBERTa, and DistilBERT sequentially, tracking progress
and generating evaluation results with bootstrap confidence intervals.
"""

import argparse
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch

sys.path.append(str(Path(__file__).parent.parent))
from src.baseline_trainer import NERTrainer
from src.reproducibility import reproducibility_block

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODELS = {
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "description": "Multilingual BERT (mBERT)",
    },
    "xlm-roberta": {
        "name": "xlm-roberta-base",
        "description": "XLM-RoBERTa Base",
    },
    "distilbert": {
        "name": "distilbert-base-multilingual-cased",
        "description": "DistilBERT Multilingual",
    },
}

PROGRESS_FILE = Path("models/baseline/training_progress.json")


def load_progress() -> Dict[str, Any]:
    """Load existing training progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "models_trained": [],
        "models_failed": [],
        "results": {},
    }


def save_progress(progress: Dict[str, Any]) -> None:
    """Save training progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def train_single_model(
    model_key: str,
    config: Dict[str, Any],
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    max_length: int = 64,
    seed: int = 42,
) -> Tuple[Dict[str, Any], bool]:
    """Train and evaluate a single transformer model."""
    logger.info("Starting training for %s (%s)", model_key, config["name"])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        trainer = NERTrainer(
            model_name=config["name"],
            data_dir=Path("data/raw"),
            output_dir=Path("models/baseline") / model_key,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
            seed=seed,
        )

        logger.info("Loading dataset splits...")
        trainer.load_data()

        logger.info("Starting training epochs...")
        trained_model = trainer.train()

        logger.info("Evaluating on held-out test split...")
        eval_results = trainer.evaluate(trained_model)

        elapsed = time.time() - start_time
        hours = elapsed / 3600
        timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result = {
            "model_name": config["name"],
            "description": config["description"],
            "training_time_hours": round(hours, 2),
            "training_time_minutes": round(elapsed / 60, 1),
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "settings": {
                "batch_size": batch_size,
                "max_length": max_length,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": seed,
            },
            "results": eval_results or {},
            "status": "success",
        }

        del trainer
        del trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Finished %s successfully in %.1f minutes.", model_key, elapsed / 60)
        return result, True

    except Exception as e:
        logger.exception("Failed training %s: %s", model_key, e)
        result = {
            "model_name": config["name"],
            "description": config["description"],
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return result, False


def save_comparison_results(progress: Dict[str, Any]) -> None:
    """Save summary of all trained models."""
    output_file = Path("results/comparison/comparison_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(progress["results"], f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NER models sequentially.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mbert", "xlm-roberta", "distilbert", "all"],
        help="Train a specific model or 'all' models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Per-device batch size (default: 64).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum sequence length (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current training progress and exit.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset training progress tracking.",
    )

    args = parser.parse_args()

    reproducibility_block(
        seed=args.seed,
        output_dir="models/baseline",
        experiment_name="sequential-baseline-training",
    )

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            logger.info("Training progress reset.")
        return

    progress = load_progress()

    if args.status:
        logger.info("Current Training Progress:")
        for key, conf in MODELS.items():
            st = "Completed" if key in progress["models_trained"] else "Failed" if key in progress["models_failed"] else "Pending"
            logger.info("  %-15s: %s", key, st)
        return

    target_models = list(MODELS.keys()) if (args.model is None or args.model == "all") else [args.model]

    for model_key in target_models:
        if model_key in progress["models_trained"]:
            logger.info("Model %s already trained. Skipping.", model_key)
            continue

        config = MODELS[model_key]
        result, success = train_single_model(
            model_key=model_key,
            config=config,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )

        if success:
            progress["models_trained"].append(model_key)
        else:
            progress["models_failed"].append(model_key)

        progress["results"][model_key] = result
        save_progress(progress)
        save_comparison_results(progress)


if __name__ == "__main__":
    main()

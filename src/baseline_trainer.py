"""
Fine-tuning and evaluation pipeline for discriminative NER models on Nepali text.

Supports mBERT, XLM-RoBERTa, and DistilBERT with seqeval metrics
and 95% bootstrap confidence intervals.
"""

import argparse
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .reproducibility import set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """PyTorch Dataset wrapper for CoNLL-formatted BIO NER data."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: Any,
        label2id: Dict[str, int],
        max_length: int = 64,
    ) -> None:
        """Initialize the NER dataset.

        Args:
            data_path: Path to the CoNLL BIO file.
            tokenizer: HuggingFace fast tokenizer.
            label2id: Mapping from entity string label to integer ID.
            max_length: Maximum sequence length for truncation and padding.
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.sentences, self.labels = self._load_conll_data(data_path)

    def _load_conll_data(
        self, file_path: Path
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Parse two-column CoNLL format into token and label lists."""
        sentences: List[List[str]] = []
        labels: List[List[str]] = []
        current_tokens: List[str] = []
        current_labels: List[str] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])

        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)

        return sentences, labels

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids: List[int] = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id.get(labels[word_idx], self.label2id["O"]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class NERTrainer:
    """Orchestrates fine-tuning and evaluation for transformer-based token classification."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        data_dir: Path = Path("data/raw"),
        output_dir: Path = Path("models/baseline"),
        max_length: int = 64,
        batch_size: int = 64,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        seed: int = 42,
    ) -> None:
        """Configure the training pipeline.

        Args:
            model_name: Pretrained model identifier on HuggingFace Hub.
            data_dir: Directory containing CoNLL BIO split files.
            output_dir: Directory where checkpoints and results will be stored.
            max_length: Maximum sequence length.
            batch_size: Batch size per device.
            learning_rate: AdamW initial learning rate.
            num_epochs: Maximum training epochs.
            seed: Random seed for deterministic reproducibility.
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed

        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label2id, self.id2label = self._get_label_mappings()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.train_dataset: Optional[NERDataset] = None
        self.val_dataset: Optional[Any] = None
        self.test_dataset: Optional[NERDataset] = None
        self.model: Optional[AutoModelForTokenClassification] = None

    def _get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Extract label mappings from the training split."""
        labels = {"O"}
        possible_files = [
            "EverestNER-train-bio.txt",
            "EverestNER-train-bio",
            "train.bio",
            "train.txt",
        ]

        train_file = None
        for filename in possible_files:
            candidate = self.data_dir / filename
            if candidate.exists():
                train_file = candidate
                break

        if train_file and train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            labels.add(parts[1])

        sorted_labels = sorted(
            labels,
            key=lambda x: (0 if x == "O" else 1 if x.startswith("B-") else 2, x),
        )

        label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        logger.info("Discovered %d entity labels: %s", len(sorted_labels), sorted_labels)
        return label2id, id2label

    def load_data(self) -> None:
        """Load and tokenize dataset splits."""
        train_file = self.data_dir / "EverestNER-train-bio.txt"
        val_file = self.data_dir / "EverestNER-val-bio.txt"
        test_file = self.data_dir / "EverestNER-test-bio.txt"

        if not train_file.exists():
            raise FileNotFoundError(f"Training split not found: {train_file}")

        self.train_dataset = NERDataset(
            train_file, self.tokenizer, self.label2id, self.max_length
        )

        if val_file.exists():
            self.val_dataset = NERDataset(
                val_file, self.tokenizer, self.label2id, self.max_length
            )
        else:
            train_size = int(0.85 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if test_file.exists():
            self.test_dataset = NERDataset(
                test_file, self.tokenizer, self.label2id, self.max_length
            )

        logger.info(
            "Dataset split sizes: Train=%d, Val=%d, Test=%s",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset) if self.test_dataset else "N/A",
        )

    def initialize_model(self) -> None:
        """Load pretrained model weights and configure classification head."""
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Loaded model %s (%s total parameters)", self.model_name, f"{total_params:,}")

    def compute_metrics(self, pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Compute strict seqeval span-level precision, recall, and F1."""
        metric = evaluate.load("seqeval")
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []

        for prediction, label in zip(predictions, labels):
            t_lbl: List[str] = []
            t_prd: List[str] = []
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    t_lbl.append(self.id2label[label_id])
                    t_prd.append(self.id2label[pred_id])
            true_labels.append(t_lbl)
            true_predictions.append(t_prd)

        results = metric.compute(predictions=true_predictions, references=true_labels)

        per_entity = {}
        for short in ["Person", "Location", "Organization", "Date", "Event"]:
            if short in results:
                per_entity[short] = {
                    "precision": float(results[short]["precision"]),
                    "recall": float(results[short]["recall"]),
                    "f1": float(results[short]["f1"]),
                    "support": int(results[short]["number"]),
                }

        return {
            "precision": float(results["overall_precision"]),
            "recall": float(results["overall_recall"]),
            "f1": float(results["overall_f1"]),
            "accuracy": float(results["overall_accuracy"]),
            "per_entity": per_entity,
        }

    def _compute_bootstrap_f1(
        self,
        all_true: List[List[str]],
        all_pred: List[List[str]],
        n_iterations: int = 500,
    ) -> Tuple[float, float, float]:
        """Compute empirical 95% confidence intervals via non-parametric bootstrapping."""
        metric = evaluate.load("seqeval")
        n_samples = len(all_true)
        indices = np.arange(n_samples)
        bootstrapped_f1s: List[float] = []

        for _ in range(n_iterations):
            batch_indices = np.random.choice(indices, size=n_samples, replace=True)
            sample_true = [all_true[i] for i in batch_indices]
            sample_pred = [all_pred[i] for i in batch_indices]
            res = metric.compute(predictions=sample_pred, references=sample_true)
            bootstrapped_f1s.append(res["overall_f1"])

        bootstrapped_f1s.sort()
        lower_bound = float(bootstrapped_f1s[int(0.025 * n_iterations)])
        upper_bound = float(bootstrapped_f1s[int(0.975 * n_iterations)])
        margin_of_error = float((upper_bound - lower_bound) / 2)

        return lower_bound, upper_bound, margin_of_error

    def train(self) -> Trainer:
        """Execute model fine-tuning with validation monitoring and early stopping."""
        if self.train_dataset is None:
            self.load_data()
        if self.model is None:
            self.initialize_model()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            seed=self.seed,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            gradient_accumulation_steps=2,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()

        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Fine-tuning completed. Saved final model to %s", final_model_path)
        return trainer

    def evaluate(self, trainer: Optional[Trainer] = None) -> Optional[Dict[str, Any]]:
        """Evaluate best checkpoint on the held-out test split."""
        if trainer is None:
            final_path = self.output_dir / "final_model"
            self.model = AutoModelForTokenClassification.from_pretrained(str(final_path))
            self.model.to(self.device)
            trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)

        if self.test_dataset is None:
            logger.warning("No test dataset available for evaluation.")
            return None

        output = trainer.predict(self.test_dataset)
        predictions = np.argmax(output.predictions, axis=2)
        labels = output.label_ids

        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []

        for prediction, label in zip(predictions, labels):
            t_lbl, t_prd = [], []
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    t_lbl.append(self.id2label[label_id])
                    t_prd.append(self.id2label[pred_id])
            true_labels.append(t_lbl)
            true_predictions.append(t_prd)

        results = self.compute_metrics((output.predictions, labels))
        low, high, margin = self._compute_bootstrap_f1(true_labels, true_predictions)

        results["f1_ci_low"] = float(low)
        results["f1_ci_high"] = float(high)
        results["f1_margin"] = float(margin)

        logger.info(
            "Evaluation on Test Set - F1: %.4f ± %.4f (95%% CI: [%.4f, %.4f])",
            results["f1"],
            margin,
            low,
            high,
        )

        results_file = self.output_dir / "test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune transformer models for Nepali NER.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Pretrained HuggingFace model identifier.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to directory containing dataset splits.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/baseline",
        help="Path to output directory.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum sequence token length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training and evaluation batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )

    args = parser.parse_args()

    trainer = NERTrainer(
        model_name=args.model_name,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir) / args.model_name.split("/")[-1],
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        seed=args.seed,
    )

    trainer.load_data()
    trained_model = trainer.train()
    trainer.evaluate(trained_model)


if __name__ == "__main__":
    main()

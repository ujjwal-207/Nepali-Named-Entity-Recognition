"""
Fine-tune Pre-trained Models for Nepali NER - CPU OPTIMIZED VERSION
Optimized for systems with 8GB RAM and NO GPU

Save this as: src/baseline_trainer.py
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import evaluate  # Replaced load_metric with evaluate
import warnings
warnings.filterwarnings('ignore')


class NERDataset(Dataset):
    """Custom Dataset for NER - Memory optimized"""
    
    def __init__(self, data_path: Path, tokenizer, label2id: Dict, max_length: int = 64):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        # Load data
        self.sentences, self.labels = self._load_conll_data(data_path)
        
    def _load_conll_data(self, file_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
        """Load CoNLL/BIO format data - DIRECT from .txt or .bio files"""
        sentences = []
        labels = []
        current_tokens = []
        current_labels = []
        
        print(f"  Loading: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line = sentence boundary
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
        
        # Don't forget last sentence
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
        
        print(f"  ✓ Loaded {len(sentences)} sentences")
        return sentences, labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]
        
        # Tokenize and align labels
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id.get(labels[word_idx], self.label2id['O']))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }


class NERTrainer:
    """Trainer for NER models - CPU & Low RAM Optimized"""
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",  # Smaller default
        data_dir: Path = Path("data/raw"),
        output_dir: Path = Path("models/baseline"),
        max_length: int = 64,  # Reduced for memory
        batch_size: int = 4,   # Small batch for CPU
        learning_rate: float = 3e-5,
        num_epochs: int = 5,   # Fewer epochs for CPU
        seed: int = 42
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Force CPU
        self.device = torch.device('cpu')
        
        print(f"\n{'='*60}")
        print(f"RUNNING ON CPU - OPTIMIZED FOR LOW MEMORY")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"RAM-friendly settings:")
        print(f"  - Batch size: {batch_size} (small for CPU)")
        print(f"  - Max length: {max_length} (reduced)")
        print(f"  - Epochs: {num_epochs} (fewer for faster training)")
        print(f"{'='*60}")
        
        # Initialize label mappings
        self.label2id, self.id2label = self._get_label_mappings()
        
        # Initialize tokenizer
        print(f"\nLoading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True  # Use fast tokenizer for speed
        )
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
        
    def _get_label_mappings(self) -> Tuple[Dict, Dict]:
        """Extract unique labels from training data"""
        labels = set(['O'])
        
        # Try different possible filenames
        possible_files = [
            "EverestNER-train-bio.txt",
            "EverestNER-train-bio",
            "train.bio",
            "train.txt"
        ]
        
        train_file = None
        for filename in possible_files:
            test_path = self.data_dir / filename
            if test_path.exists():
                train_file = test_path
                break
        
        if train_file and train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            labels.add(parts[1])
        
        sorted_labels = sorted(labels, key=lambda x: (
            0 if x == 'O' else 1 if x.startswith('B-') else 2, x
        ))
        
        label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        print(f"\n{'='*60}")
        print(f"Entity Labels: {len(sorted_labels)}")
        print(f"{'='*60}")
        print(f"Labels: {sorted_labels}")
        
        return label2id, id2label
    
    def load_data(self):
        """Load train, validation, test datasets from BIO files"""
        print(f"\n{'='*60}")
        print("Loading Datasets (BIO format)")
        print(f"{'='*60}")
        print("NOTE: Training uses ORIGINAL BIO/TXT files, NOT JSON!")
        print(f"{'='*60}")
        
        # Try different possible filenames for train
        possible_train_files = [
            "EverestNER-train-bio",
            "EverestNER-train-bio.txt",
            "train.bio",
            "train.txt"
        ]
        
        train_file = None
        for filename in possible_train_files:
            test_path = self.data_dir / filename
            if test_path.exists():
                train_file = test_path
                break
        
        if not train_file:
            raise FileNotFoundError(
                f"\n❌ Training file not found in: {self.data_dir}\n"
                f"Tried: {', '.join(possible_train_files)}\n"
                f"Please ensure your BIO format file is in this location.\n"
                f"Format: token TAG (one per line, empty line between sentences)"
            )
        
        self.train_dataset = NERDataset(
            train_file, self.tokenizer, self.label2id, self.max_length
        )
        
        # Validation
        val_file = self.data_dir / "EverestNER-val-bio"
        if not val_file.exists():
            val_file = self.data_dir / "EverestNER-val-bio.txt"
        
        if val_file.exists():
            self.val_dataset = NERDataset(
                val_file, self.tokenizer, self.label2id, self.max_length
            )
        else:
            print("\n  No validation file. Creating 90/10 split from training data...")
            train_size = int(0.9 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
        
        # Test (optional)
        test_file = self.data_dir / "EverestNER-test-bio"
        if not test_file.exists():
            test_file = self.data_dir / "EverestNER-test-bio.txt"
        
        if test_file.exists():
            self.test_dataset = NERDataset(
                test_file, self.tokenizer, self.label2id, self.max_length
            )
        
        print(f"\n{'='*60}")
        print(f"Dataset Sizes:")
        print(f"{'='*60}")
        print(f"  Train:      {len(self.train_dataset):,}")
        print(f"  Validation: {len(self.val_dataset):,}")
        if self.test_dataset:
            print(f"  Test:       {len(self.test_dataset):,}")
        print(f"{'='*60}")
    
    def initialize_model(self):
        """Initialize model - optimized for CPU"""
        print(f"\n{'='*60}")
        print(f"Downloading & Initializing Model: {self.model_name}")
        print(f"{'='*60}")
        print("This will download the model to your PC (~500MB)")
        print("Model will be cached for future use")
        print(f"{'='*60}")
        
        # Download and initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        
        # Clear memory
        gc.collect()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n✓ Model downloaded and loaded")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Number of labels: {len(self.label2id)}")
        print(f"{'='*60}")
    
    def compute_metrics(self, pred):
        """Compute F1, precision, recall"""
        # Use evaluate instead of datasets.load_metric
        metric = evaluate.load("seqeval")
        
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)
        
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(predictions, labels):
            true_label = []
            true_prediction = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_label.append(self.id2label[label_id])
                    true_prediction.append(self.id2label[pred_id])
            
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def train(self):
        """Train the model - CPU optimized"""
        if self.train_dataset is None:
            self.load_data()
        
        if self.model is None:
            self.initialize_model()
        
        print(f"\n{'='*60}")
        print("Starting Training on CPU")
        print(f"{'='*60}")
        print(f"⚠️  WARNING: CPU training is SLOW!")
        print(f"  Expected time: 4-8 hours for {self.num_epochs} epochs")
        print(f"\nSettings:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max length: {self.max_length}")
        print(f"{'='*60}\n")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CPU-optimized training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=100,  # Reduced warmup
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,  # Keep only 2 checkpoints to save space
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none",  # Disable reporting for speed
            seed=self.seed,
            fp16=False,  # No mixed precision on CPU
            dataloader_num_workers=0,  # No parallel workers
            gradient_accumulation_steps=2,  # Accumulate gradients
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("🚀 Training started...")
        print("💡 TIP: This will take several hours. You can:")
        print("   - Leave it running overnight")
        print("   - Use --epochs 3 for faster testing")
        print("   - Check progress in the logs folder\n")
        
        # Train
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Clear memory
        gc.collect()
        
        print(f"\n{'='*60}")
        print("✓ Training Complete!")
        print(f"{'='*60}")
        print(f"Model saved to: {final_model_path}")
        print(f"Model size: ~500MB")
        print(f"{'='*60}\n")
        
        return trainer
    
    def evaluate(self, trainer: Optional[Trainer] = None):
        """Evaluate on test set"""
        if trainer is None:
            print("Loading trained model for evaluation...")
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(self.output_dir / "final_model")
            )
            self.model.to(self.device)
            
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(output_dir=str(self.output_dir)),
                compute_metrics=self.compute_metrics,
            )
        
        if self.test_dataset is not None:
            print(f"\n{'='*60}")
            print("Evaluating on Test Set")
            print(f"{'='*60}")
            
            results = trainer.evaluate(self.test_dataset)
            
            print(f"\n{'='*60}")
            print("Test Results")
            print(f"{'='*60}")
            print(f"  Precision: {results['eval_precision']:.4f} ({results['eval_precision']*100:.2f}%)")
            print(f"  Recall:    {results['eval_recall']:.4f} ({results['eval_recall']*100:.2f}%)")
            print(f"  F1 Score:  {results['eval_f1']:.4f} ({results['eval_f1']*100:.2f}%)")
            print(f"  Accuracy:  {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
            print(f"{'='*60}\n")
            
            # Save results
            results_file = self.output_dir / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✓ Results saved to: {results_file}\n")
            return results
        else:
            print("\n⚠ No test dataset available")
            return None


def main():
    """Main training pipeline - CPU optimized"""
    print("\n" + "="*70)
    print("NEPALI NER BASELINE - CPU & LOW RAM VERSION")
    print("="*70)
    print("\n⚠️  IMPORTANT NOTES:")
    print("  1. Training uses ORIGINAL BIO/TXT files (NOT JSON)")
    print("  2. Model will be downloaded to your PC (~500MB)")
    print("  3. CPU training takes 4-8 hours")
    print("  4. Optimized for 8GB RAM")
    print("="*70 + "\n")
    
    # Initialize trainer with smaller, faster model
    trainer = NERTrainer(
        model_name="bert-base-multilingual-cased",  # Good balance for CPU
        data_dir=Path("data/raw"),
        output_dir=Path("models/baseline/bert-base-multilingual-cased"),
        max_length=64,    # Reduced for memory
        batch_size=4,     # Small batch for CPU
        learning_rate=3e-5,
        num_epochs=5,     # Fewer epochs
        seed=42
    )
    
    # Full pipeline
    trainer.load_data()
    trained_model = trainer.train()
    trainer.evaluate(trained_model)
    
    print("\n" + "="*70)
    print("✓ ALL DONE!")
    print("="*70)
    print("\nYour trained model is saved at:")
    print(f"  models/baseline/bert-base-multilingual-cased/final_model/")
    print("\nYou can now use it for predictions!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
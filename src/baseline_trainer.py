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
import evaluate 
import warnings
warnings.filterwarnings('ignore')


class NERDataset(Dataset):
    
    def __init__(self, data_path: Path, tokenizer, label2id: Dict, max_length: int = 64):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        self.sentences, self.labels = self._load_conll_data(data_path)
        
    def _load_conll_data(self, file_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
        sentences = []
        labels = []
        current_tokens = []
        current_labels = []
        
        print(f"  Loading: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
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
        
        print(f"  Loaded {len(sentences)} sentences")
        return sentences, labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) 
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
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased", 
        data_dir: Path = Path("data/raw"),
        output_dir: Path = Path("models/baseline"),
        max_length: int = 64,
        batch_size: int = 10,   
        learning_rate: float = 3e-5,
        num_epochs: int = 5,  
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
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.label2id, self.id2label = self._get_label_mappings()
        
        print(f"\n tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True 
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
    def _compute_bootstrap_f1(self, all_true, all_pred, n_iterations=500):
        
        import evaluate
        metric = evaluate.load("seqeval")
        bootstrapped_f1s = []
        
        print(f"  Calculating 95% CI (Bootstrapping {n_iterations} iterations)...")
        
        n_samples = len(all_true)
        indices = np.arange(n_samples)
        
        for _ in range(n_iterations):
            # Sample with replacement
            batch_indices = np.random.choice(indices, size=n_samples, replace=True)
            sample_true = [all_true[i] for i in batch_indices]
            sample_pred = [all_pred[i] for i in batch_indices]
            
            res = metric.compute(predictions=sample_pred, references=sample_true)
            bootstrapped_f1s.append(res["overall_f1"])
            
        bootstrapped_f1s.sort()
        # 95% CI is between 2.5th and 97.5th percentile
        lower_bound = bootstrapped_f1s[int(0.025 * n_iterations)]
        upper_bound = bootstrapped_f1s[int(0.975 * n_iterations)]
        margin_of_error = (upper_bound - lower_bound) / 2
        
        return lower_bound, upper_bound, margin_of_error    
    def _get_label_mappings(self) -> Tuple[Dict, Dict]:
        labels = set(['O'])
        
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
        
        print(f"Entity Labels: {len(sorted_labels)}")
        print(f"Labels: {sorted_labels}")
        
        return label2id, id2label
    
    def load_data(self):
        
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
                f"\nfile not found in: {self.data_dir}\n"
                f"Tried: {', '.join(possible_train_files)}\n"
            )
        
        self.train_dataset = NERDataset(
            train_file, self.tokenizer, self.label2id, self.max_length
        )
        
        val_file = self.data_dir / "EverestNER-val-bio"
        if not val_file.exists():
            val_file = self.data_dir / "EverestNER-val-bio.txt"
        
        if val_file.exists():
            self.val_dataset = NERDataset(
                val_file, self.tokenizer, self.label2id, self.max_length
            )
        else:
            train_size = int(0.9 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
        
        test_file = self.data_dir / "EverestNER-test-bio"
        if not test_file.exists():
            test_file = self.data_dir / "EverestNER-test-bio.txt"
        
        if test_file.exists():
            self.test_dataset = NERDataset(
                test_file, self.tokenizer, self.label2id, self.max_length
            )
        
        print(f"  Train:      {len(self.train_dataset):,}")
        print(f"  Validation: {len(self.val_dataset):,}")
        if self.test_dataset:
            print(f"  Test:       {len(self.test_dataset):,}")
        print(f"{'='*60}")
    
    def initialize_model(self):
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        
        gc.collect()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n model downloaded and loaded")
        print(f"  Total parameters: {total_params:,}")
    
    def compute_metrics(self, pred):
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
        
        # Build per-entity metrics
        per_entity = {}
        # Collect per-entity metrics from seqeval
        for short in ["Person", "Location", "Organization", "Date", "Event"]:
                if short in results:
                    per_entity[short] = {
                        "precision": float(results[short]["precision"]),
                        "recall": float(results[short]["recall"]),
                        "f1": float(results[short]["f1"]),
                        "support": int(results[short]["number"])
                    }
        
        return {
            "precision": float(results["overall_precision"]),
            "recall": float(results["overall_recall"]),
            "f1": float(results["overall_f1"]),
            "accuracy": float(results["overall_accuracy"]),
            "per_entity": per_entity if per_entity else None
        }
    
    def train(self):
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
            warmup_steps=100, 
            # logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none", 
            seed=self.seed,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  
            gradient_accumulation_steps=2,  
        )
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        
        trainer.train()
        
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        gc.collect()
        
        print(" Training Complete!")
        print(f"Model saved to: {final_model_path}")
        
        return trainer
    
    def evaluate(self, trainer: Optional[Trainer] = None):
        if trainer is None:
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.output_dir / "final_model"))
            self.model.to(self.device)
            trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)

        if self.test_dataset is not None:
            # 1. Get raw predictions
            output = trainer.predict(self.test_dataset)
            predictions = np.argmax(output.predictions, axis=2)
            labels = output.label_ids

            true_labels = []
            true_predictions = []

            for prediction, label in zip(predictions, labels):
                t_lbl, t_prd = [], []
                for pred_id, label_id in zip(prediction, label):
                    if label_id != -100:
                        t_lbl.append(self.id2label[label_id])
                        t_prd.append(self.id2label[pred_id])
                true_labels.append(t_lbl)
                true_predictions.append(t_prd)

            # 2. Calculate Standard Metrics
            results = self.compute_metrics((output.predictions, labels))
            
            # 3. Calculate Confidence Intervals
            low, high, margin = self._compute_bootstrap_f1(true_labels, true_predictions)
            results['f1_ci_low'] = float(low)
            results['f1_ci_high'] = float(high)
            results['f1_margin'] = float(margin)

            print(f"  F1 Score:  {results['f1']:.4f} ± {margin:.4f} (95% CI: [{low:.4f}, {high:.4f}])")
            
            # Print per-entity breakdown
            if results.get('per_entity'):
                print(f"{'='*60}")
                print(f"  Per-Entity Breakdown:")
                print(f"  {'Entity':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<8}")
                print(f"  {'-'*50}")
                for entity, metrics in results['per_entity'].items():
                    print(f"  {entity:<8} {metrics['precision']:.4f}       {metrics['recall']:.4f}       {metrics['f1']:.4f}       {metrics['support']:<8}")
                print(f"{'='*60}")
            
            # Save results
            with open(self.output_dir / "test_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
        else:
            print("\n test dataset not available")
            return None


def main():
    
    trainer = NERTrainer(
        model_name="bert-base-multilingual-cased", 
        data_dir=Path("data/raw"),
        output_dir=Path("models/baseline/bert-base-multilingual-cased"),
        max_length=64,  
        batch_size=4,     
        learning_rate=3e-5,
        num_epochs=5,   
        seed=42
    )
    
    trainer.load_data()
    trained_model = trainer.train()
    trainer.evaluate(trained_model)
    
    print("done")


if __name__ == "__main__":
    main()
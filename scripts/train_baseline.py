
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
        
        print(f"   Loaded {len(sentences)} sentences")
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
        batch_size: int = 4, 
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
        
        self.device = torch.device('cpu')
        
        
        self.label2id, self.id2label = self._get_label_mappings()
        
        print(f"\nLoading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True 
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
        
    def _get_label_mappings(self) -> Tuple[Dict, Dict]:
        labels = set(['O'])
        
        possible_files = [
            "EverestNER-train-bio",
            "EverestNER-train-bio.txt",
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
        print("Loading Datasets (BIO format)")
        
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
                f"\n Training file not found in: {self.data_dir}\n"
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
            print("\n  No validation file. Creating 90/10 split from training data...")
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
        
        print(f"Dataset Sizes:")
        print(f"  Train:      {len(self.train_dataset):,}")
        print(f"  Validation: {len(self.val_dataset):,}")
        if self.test_dataset:
            print(f"  Test:       {len(self.test_dataset):,}")
    
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
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Number of labels: {len(self.label2id)}")
    
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
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
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
            logging_dir=str(self.output_dir / 'logs'),
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
            fp16=False, 
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
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        
        trainer.train()
        
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        gc.collect()
        
        
        return trainer
    
    def evaluate(self, trainer: Optional[Trainer] = None):
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
            
            results = trainer.evaluate(self.test_dataset)
            
            print(f"  Precision: {results['eval_precision']:.4f} ({results['eval_precision']*100:.2f}%)")
            print(f"  Recall:    {results['eval_recall']:.4f} ({results['eval_recall']*100:.2f}%)")
            print(f"  F1 Score:  {results['eval_f1']:.4f} ({results['eval_f1']*100:.2f}%)")
            print(f"  Accuracy:  {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
            
            results_file = self.output_dir / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
        else:
            print("\n No test dataset available")
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
    


if __name__ == "__main__":
    main()

""" i dont think this will work in my local pc so let's just try  """
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json
import argparse
from src.reproducibility import reproducibility_block
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')


MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",  # Smaller Llama
        "description": "Llama 3.2 3B (Meta) - Good balance",
        "context_length": 8192,
        "requires_token": True  
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct - Excellent performance",
        "context_length": 8192,
        "requires_token": False
    },
    "gemma": {
        "name": "google/gemma-2b-it",
        "description": "Gemma 2B (Google) - Fastest, lowest RAM",
        "context_length": 8192,
        "requires_token": True
    },
    "phi": {
        "name": "microsoft/phi-2",
        "description": "Phi-2 (Microsoft) - 2.7B, very efficient",
        "context_length": 2048,
        "requires_token": False
    }
}


class NERDatasetQLoRA:
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences, self.labels = self._load_conll_data(data_path)
        
    def _load_conll_data(self, file_path: Path):
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
    
    def _format_instruction(self, tokens, labels):
        entities = []
        current_entity = None
        current_text = []
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_entity
                    })
                current_entity = label[2:]
                current_text = [token]
            elif label.startswith('I-') and current_entity:
                current_text.append(token)
            else:
                if current_entity:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_entity
                    })
                    current_entity = None
                    current_text = []
        
        if current_entity:
            entities.append({
                "text": " ".join(current_text),
                "type": current_entity
            })
        
        sentence = " ".join(tokens)
        
        instruction = f"""Extract all named entities from the following Nepali text. Identify entities such as PERSON, LOCATION, ORGANIZATION, DATE, TIME, etc.

Text: {sentence}

Entities:"""
        
        if entities:
            entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
        else:
            entities_json = "[]"
        
        response = entities_json
        
        return instruction, response
    
    def create_dataset(self):
        data = []
        
        for tokens, labels in zip(self.sentences, self.labels):
            instruction, response = self._format_instruction(tokens, labels)
            
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
            
            data.append({
                "text": prompt,
                "instruction": instruction,
                "response": response
            })
        
        return Dataset.from_list(data)


class QLoRANERTrainer:
    """QLoRA trainer for NER with generative models"""
    
    def __init__(
        self,
        model_key: str = "phi",
        data_dir: Path = Path("data/raw"),
        instruction_dir: Path = Path("data/instruction_format"),
        output_dir: Path = Path("models/qlora"),
        max_length: int = 512,
        batch_size: int = 2,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        hf_token: str = None,
        use_instruction_files: bool = False
    ):
        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.model_name = self.model_config["name"]
        self.data_dir = data_dir
        self.instruction_dir = instruction_dir
        self.output_dir = output_dir / model_key
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hf_token = hf_token
        self.use_instruction_files = use_instruction_files
        
        
    def setup_model_and_tokenizer(self):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.hf_token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Loading model: {self.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16, 
            max_memory={0: "6GB", "cpu": "6GB"} 
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        print(" Model loaded with 4-bit quantization")
        
    def setup_lora(self):
        
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n LoRA setup complete")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Total params: {total_params:,}")
        print(f"  Memory saved: ~{100*(1-trainable_params/total_params):.1f}%\n")
        
    def load_data(self):
        
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
        
        if not train_file:
            raise FileNotFoundError(f"Training file not found in {self.data_dir}")
        
        print("Converting to instruction format...")
        train_loader = NERDatasetQLoRA(train_file, self.tokenizer, self.max_length)
        self.train_dataset = train_loader.create_dataset()
        
        val_file = self.data_dir / "EverestNER-val-bio"
        if not val_file.exists():
            val_file = self.data_dir / "EverestNER-val-bio.txt"
        
        if val_file.exists():
            val_loader = NERDatasetQLoRA(val_file, self.tokenizer, self.max_length)
            self.val_dataset = val_loader.create_dataset()
        else:
            print("No validation file, using 90/10 split")
            split = self.train_dataset.train_test_split(test_size=0.1, seed=42)
            self.train_dataset = split['train']
            self.val_dataset = split['test']
        
        print(f"\n{'='*70}")
        print(f"Dataset Sizes:")
        print(f"  Train: {len(self.train_dataset):,}")
        print(f"  Val:   {len(self.val_dataset):,}")
        print(f"{'='*70}\n")
        
    def tokenize_function(self, examples):
        outputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    def train(self):
        if not hasattr(self, 'model'):
            self.setup_model_and_tokenizer()
            self.setup_lora()
        
        if not hasattr(self, 'train_dataset'):
            self.load_data()
        
        
        train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        val_dataset = self.val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = 8
            learning_rate=self.learning_rate,
            warmup_steps=50,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True, 
            optim="paged_adamw_8bit",
            report_to="none",
            seed=42,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        
        trainer.train()
        
        print("\nSaving model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        results = {
            "model": self.model_name,
            "model_key": self.model_key,
            "method": "QLoRA (4-bit)",
            "epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(" Training Complete!")
        print(f"Model saved to: {self.output_dir / 'final_model'}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return trainer


def main():
    reproducibility_block(seed=42, output_dir="models/qlora", experiment_name="qlora-finetuning")
    parser = argparse.ArgumentParser(description='Train generative models for NER using QLoRA')
    parser.add_argument('--model', type=str, default='phi',
                       choices=list(MODELS.keys()),
                       help='Model to train (phi/mistral/llama/gemma)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--token', type=str, help='HuggingFace token (for gated models)')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        for key, config in MODELS.items():
            token_req = "Requires HF Token" if config['requires_token'] else " No token needed"
            print(f"{key:<10} - {config['description']:<40} {token_req}")
        return
    
    model_config = MODELS[args.model]
    
    
    if model_config['requires_token'] and not args.token:
        return
    
    trainer = QLoRANERTrainer(
        model_key=args.model,
        data_dir=Path("data/raw"),
        output_dir=Path("models/qlora"),
        max_length=512,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        hf_token=args.token
    )
    
    trainer.train()
    

if __name__ == "__main__":
    main()

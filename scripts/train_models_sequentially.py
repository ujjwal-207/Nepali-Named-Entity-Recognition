import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.baseline_trainer import NERTrainer
import json
import time
import gc
import torch
import argparse
from src.reproducibility import reproducibility_block
from datetime import datetime

MODELS = {
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "batch_size": 64,
        "max_length": 64,
        "epochs": 10,
        "description": "Multilingual BERT (Google)",
        "ram_usage": "4-5GB",
        "training_time": "5-6 hours"
    },
    "xlm-roberta": {
        "name": "xlm-roberta-base", 
        "batch_size": 64,
        "max_length": 64,
        "epochs": 10,
        "description": "XLM-RoBERTa Base (Facebook)",
        "ram_usage": "6-7GB",
        "training_time": "12-15 hours"
    },
    "distilbert": {
        "name": "distilbert-base-multilingual-cased",
        "batch_size": 64,
        "max_length": 64,
        "epochs": 10,
        "description": "DistilBERT Multilingual (HuggingFace)",
        "ram_usage": "3-4GB",
        "training_time": "4-5 hours"
    }
}

PROGRESS_FILE = Path("models/baseline/training_progress.json")


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "models_trained": [],
        "models_failed": [],
        "results": {}
    }


def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def train_single_model(model_key, config):
    
    print(f"TRAINING: {config['description']}")
    print(f"Model: {config['name']}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        trainer = NERTrainer(
            model_name=config['name'],
            data_dir=Path("data/raw"),
            output_dir=Path("models/baseline") / model_key,
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            learning_rate=3e-5,
            num_epochs=config['epochs'],
            seed=42
        )
        
        print("Loading data...")
        trainer.load_data()
        
        print("\nStarting training...")
        trained_model = trainer.train()
        
        print("\nEvaluating model...")
        eval_results = trainer.evaluate(trained_model)
        
        training_time = time.time() - start_time
        hours = training_time / 3600
        timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = {
            "model_name": config['name'],
            "description": config['description'],
            "training_time_hours": round(hours, 2),
            "training_time_minutes": round(training_time / 60, 1),
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "settings": {
                "batch_size": config['batch_size'],
                "max_length": config['max_length'],
                "epochs": config['epochs']
            },
            "results": eval_results if eval_results else {},
            "status": "success"
        }
        
        print(f"{model_key.upper()} COMPLETE!")
        print(f"Time taken: {hours:.2f} hours ({training_time/60:.1f} minutes)")
        if eval_results:
            print(f"\nResults:")
            print(f"  F1 Score:  {eval_results.get('eval_f1', 0):.4f} ({eval_results.get('eval_f1', 0)*100:.2f}%)")
            print(f"  Precision: {eval_results.get('eval_precision', 0):.4f}")
            print(f"  Recall:    {eval_results.get('eval_recall', 0):.4f}")
        
        del trainer
        del trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result, True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f" {model_key.upper()} FAILED: OUT OF MEMORY")
            
            result = {
                "model_name": config['name'],
                "description": config['description'],
                "status": "failed_oom",
                "error": "Out of memory - requires more than 8GB RAM",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result, False
        else:
            raise
    
    except Exception as e:
        print(f" {model_key.upper()} FAILED")
        print(f"Error: {e}")
        
        result = {
            "model_name": config['name'],
            "description": config['description'],
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return result, False


def show_status():
    progress = load_progress()
    for model_key, config in MODELS.items():
        if model_key in progress['models_trained']:
            result = progress['results'].get(model_key, {})
            stats = result.get('results', {})
            f1 = stats.get('f1', 0)
            margin = stats.get('f1_margin', 0)
            # Display F1 with Margin
            f1_str = f"{f1:.4f} (±{margin:.4f})"
            time_str = f"{result.get('training_time_hours', 0):.1f}h"
            status = "✓ Complete"
        elif model_key in progress['models_failed']:
            f1_str = "N/A"
            time_str = "N/A"
            status = " Failed"
        else:
            f1_str = "N/A"
            time_str = "N/A"
            status = " Pending"
        
        print(f"{model_key:<20} {status:<15} {f1_str:<15} {time_str:<15}")
    
    
    total = len(MODELS)
    completed = len(progress['models_trained'])
    failed = len(progress['models_failed'])
    remaining = total - completed - failed
    
    print(f"Summary:")
    print(f"  Total models:{total}")
    print(f" Completed:{completed}")
    print()


def print_next_model_to_train():
    progress = load_progress()
    trained = set(progress['models_trained'])
    failed = set(progress['models_failed'])
    
    for model_key, config in MODELS.items():
        if model_key not in trained and model_key not in failed:
            return model_key
    
    return None


def save_comparison_results(progress):
    
    output_file = Path("models/baseline/comparison_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(progress['results'], f, indent=2)
    
    summary_file = Path("models/baseline/comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for model_key, result in progress['results'].items():
            f.write(f"\n{model_key.upper()}: {result.get('description', 'N/A')}\n")
            
            if result.get('status') == 'success':
                f.write(f"Status: Success\n")
                f.write(f"Training Time: {result.get('training_time_hours', 0):.2f} hours\n")
                f.write(f"Training Period: {result.get('timestamp_start', 'N/A')} to {result.get('timestamp_end', 'N/A')}\n")
                
                eval_results = result.get('results', {})
                if eval_results:
                    f.write(f"  F1 Score:  {eval_results.get('f1', 0):.4f} ± {eval_results.get('f1_margin', 0):.4f} ({eval_results.get('f1', 0)*100:.2f}%)\n")
                    f.write(f"  95% CI:    [{eval_results.get('f1_ci_low', 0):.4f}, {eval_results.get('f1_ci_high', 0):.4f}] ({eval_results.get('f1_ci_low', 0)*100:.2f}% - {eval_results.get('f1_ci_high', 0)*100:.2f}%)\n")
                    f.write(f"  Precision: {eval_results.get('eval_precision', 0):.4f} ({eval_results.get('eval_precision', 0)*100:.2f}%)\n")
                    f.write(f"  Recall:    {eval_results.get('eval_recall', 0):.4f} ({eval_results.get('eval_recall', 0)*100:.2f}%)\n")
                    f.write(f"  Accuracy:  {eval_results.get('eval_accuracy', 0):.4f} ({eval_results.get('eval_accuracy', 0)*100:.2f}%)\n")
            else:
                f.write(f"Status: {result.get('status', 'Failed')}\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n")
    
    print(f"\n✓ Comparison results saved:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}\n")


def main():
    reproducibility_block(seed=42, output_dir="models/baseline", experiment_name="sequential-training")
    parser = argparse.ArgumentParser(description='Train NER models sequentially')
    parser.add_argument('--model', type=str, help='Train specific model (mbert/xlm-roberta/distilbert)')
    parser.add_argument('--status', action='store_true', help='Show training status')
    parser.add_argument('--reset', action='store_true', help='Reset progress (start over)')
    
    args = parser.parse_args()
    
    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            print("\n Progress reset. Starting fresh.\n")
        else:
            print("\n No progress file found. Already clean.\n")
        return
    
    if args.status:
        show_status()
        print_next_model_to_train()
        return
    
    progress = load_progress()
    
    if args.model:
        model_key = args.model.lower()
        if model_key not in MODELS:
            print(f"\n Unknown model: {model_key}")
            print(f"Available models: {', '.join(MODELS.keys())}\n")
            return
        
        if model_key in progress['models_trained']:
            response = input("Train again anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Exiting...\n")
                return
            progress['models_trained'].remove(model_key)
    else:
        
        show_status()
        
        trained = set(progress['models_trained'])
        failed = set(progress['models_failed'])
        remaining = [k for k in MODELS.keys() if k not in trained and k not in failed]
        
        
        print("Which model do you want to train?")
        for i, model_key in enumerate(remaining, 1):
            config = MODELS[model_key]
            print(f"  {i}. {model_key:<15} - {config['description']:<35} (~{config['training_time']})")
        print(f"  0. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == '0':
            print("Exiting...\n")
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(remaining):
                model_key = remaining[idx]
            else:
                print("Invalid choice. Exiting.\n")
                return
        except ValueError:
            print("Invalid choice. Exiting.\n")
            return
    
    config = MODELS[model_key]
    
    
    response = input("Start training? (yes/no): ")
    if response.lower() != 'yes':
        print("Exiting...\n")
        return
    
    result, success = train_single_model(model_key, config)
    
    if success:
        if model_key not in progress['models_trained']:
            progress['models_trained'].append(model_key)
        if model_key in progress['models_failed']:
            progress['models_failed'].remove(model_key)
    else:
        if model_key not in progress['models_failed']:
            progress['models_failed'].append(model_key)
    
    progress['results'][model_key] = result
    save_progress(progress)
    
    save_comparison_results(progress)
    
    
    show_status()
    print_next_model_to_train()
    
    print(" Results saved:")


if __name__ == "__main__":
    main()
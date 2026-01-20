
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.baseline_trainer import NERTrainer
import json


def evaluate_model(model_dir: Path, model_key: str):
    
    
    final_model_dir = model_dir / "final_model"
    if not final_model_dir.exists():
        print(f" Model not found at {final_model_dir}")
        print(f"Skipping {model_key}...")
        return None
    
    model_names = {
        "mbert": "bert-base-multilingual-cased",
        "distilbert": "distilbert-base-multilingual-cased",
        "xlm-roberta": "xlm-roberta-base"
    }
    
    model_name = model_names.get(model_key)
    if not model_name:
        print(f" Unknown model key: {model_key}")
        return None
    
    try:
        trainer = NERTrainer(
            model_name=model_name,
            data_dir=Path("data/raw"),
            output_dir=model_dir,
            max_length=64,
            batch_size=4,
            seed=42
        )
        
        trainer.load_data()
        
        if trainer.test_dataset is None:
            print(" No test dataset found!")
            return None
        
        from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
        
        model = AutoModelForTokenClassification.from_pretrained(str(final_model_dir))
        model.to(trainer.device)
        
        eval_trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(model_dir),
                per_device_eval_batch_size=4
            ),
            compute_metrics=trainer.compute_metrics,
        )
        
        results = eval_trainer.evaluate(trainer.test_dataset)
        
        
        results_file = model_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        
        return results
        
    except Exception as e:
        print(f" Error evaluating {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_comparison_results(model_key: str, eval_results: dict):
    
    comparison_file = Path("models/baseline/comparison_results.json")
    
    if not comparison_file.exists():
        print(f" Comparison file not found: {comparison_file}")
        return
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    if model_key in data:
        data[model_key]['results'] = eval_results
        data[model_key]['status'] = 'success'
        
        with open(comparison_file, 'w') as f:
            json.dump(data, f, indent=2)
        
    else:
        print(f"  {model_key} not found in comparison results")


def main():
    
    
    base_dir = Path("models/baseline")
    
    models = {
        "mbert": base_dir / "mbert",
        "distilbert": base_dir / "distilbert",
        "xlm-roberta": base_dir / "xlm-roberta"
    }
    
    existing_models = {}
    for key, path in models.items():
        if (path / "final_model").exists():
            existing_models[key] = path
            print(f" Found: {key}")
        else:
            print(f"  Not found: {key}")
    
    if not existing_models:
        print("\n No trained models found!")
        return
    
    
    results_summary = {}
    
    for model_key, model_dir in existing_models.items():
        results = evaluate_model(model_dir, model_key)
        
        if results:
            results_summary[model_key] = results
            update_comparison_results(model_key, results)
    
    
    for model_key, results in results_summary.items():
        f1 = results.get('eval_f1', 0)
        precision = results.get('eval_precision', 0)
        recall = results.get('eval_recall', 0)
        
        print(f"{model_key.upper():<20} {f1:.4f}      {precision:.4f}      {recall:.4f}")
    
   


if __name__ == "__main__":
    main()

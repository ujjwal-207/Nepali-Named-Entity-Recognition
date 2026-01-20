
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List


class ResultsComparator:
    
    def __init__(self):
        self.bert_results = {}
        self.llm_results = {}
        
    def load_bert_results(self):
        
        comparison_file = Path("models/baseline/comparison_results.json")
        
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                data = json.load(f)
                
            for model_key, result in data.items():
                if result.get('status') == 'success':
                    metrics = result.get('results', {})
                    self.bert_results[model_key] = {
                        'name': result.get('description', model_key),
                        'precision': metrics.get('eval_precision', 0),
                        'recall': metrics.get('eval_recall', 0),
                        'f1': metrics.get('eval_f1', 0),
                        'training_time': result.get('training_time_hours', 0),
                        'cost': 0,  # Free (local)
                        'type': 'Fine-tuned'
                    }
            
            print(f"✓ Loaded {len(self.bert_results)} fine-tuned model results")
        else:
            print(f" No BERT results found at {comparison_file}")
    
    def load_llm_results(self):
        
        results_dir = Path("results/llm_api")
        
        if not results_dir.exists():
            print(f"No LLM API results found at {results_dir}")
            return
        
        for summary_file in results_dir.glob("*_summary.json"):
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            model_name = data['model']
            shots = data['num_shots']
            key = f"{model_name}_{shots}shot"
            
            self.llm_results[key] = {
                'name': f"{model_name.upper()} ({shots}-shot)",
                'precision': data['metrics']['precision'],
                'recall': data['metrics']['recall'],
                'f1': data['metrics']['f1'],
                'training_time': 0,  # No training
                'cost': data['costs']['cost_per_1k_samples'],
                'latency': data['performance']['avg_time_ms'],
                'type': 'LLM API'
            }
        
        print(f"✓ Loaded {len(self.llm_results)} LLM API results")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison DataFrame"""
        
        all_results = []
        
        for key, data in self.bert_results.items():
            all_results.append({
                'Model': data['name'],
                'Type': data['type'],
                'Precision': data['precision'],
                'Recall': data['recall'],
                'F1 Score': data['f1'],
                'Training Time (hrs)': data['training_time'],
                'Cost per 1K': f"$0.00",
                'Inference (ms)': 'Fast (~50ms)',
                'Setup': 'Complex'
            })
        for key, data in self.llm_results.items():
            all_results.append({
                'Model': data['name'],
                'Type': data['type'],
                'Precision': data['precision'],
                'Recall': data['recall'],
                'F1 Score': data['f1'],
                'Training Time (hrs)': 0,
                'Cost per 1K': f"${data['cost']:.2f}",
                'Inference (ms)': f"{data.get('latency', 0):.0f}ms",
                'Setup': 'Simple'
            })
        
        df = pd.DataFrame(all_results)
        df = df.sort_values('F1 Score', ascending=False)
        
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """Print formatted comparison table"""
        print("\n" + "="*100)
        print("COMPLETE COMPARISON: FINE-TUNED vs LLM APIs")
        print("="*100 + "\n")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)
        pd.set_option('display.max_colwidth', 30)
        
        print(df.to_string(index=False))
        print("\n" + "="*100)
    
    def create_latex_table(self, df: pd.DataFrame) -> str:
        
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Comparison of Fine-tuned Models and LLM APIs for Nepali NER}\n"
        latex += "\\label{tab:comparison}\n"
        latex += "\\begin{tabular}{lccccc}\n"
        latex += "\\hline\n"
        latex += "Model & F1 Score & Precision & Recall & Training Time & Cost/1K \\\\\n"
        latex += "\\hline\n"
        
        for _, row in df.iterrows():
            latex += f"{row['Model']} & "
            latex += f"{row['F1 Score']:.4f} & "
            latex += f"{row['Precision']:.4f} & "
            latex += f"{row['Recall']:.4f} & "
            latex += f"{row['Training Time (hrs)']:.1f}h & "
            latex += f"{row['Cost per 1K']} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def create_markdown_table(self, df: pd.DataFrame) -> str:
        """Create Markdown table for paper"""
        
        markdown = "## Complete Results Comparison\n\n"
        markdown += "| Model | Type | F1 Score | Precision | Recall | Training | Cost/1K |\n"
        markdown += "|-------|------|----------|-----------|--------|----------|----------|\n"
        
        for _, row in df.iterrows():
            markdown += f"| {row['Model']:<25} | "
            markdown += f"{row['Type']:<12} | "
            markdown += f"{row['F1 Score']:.4f} | "
            markdown += f"{row['Precision']:.4f} | "
            markdown += f"{row['Recall']:.4f} | "
            markdown += f"{row['Training Time (hrs)']:.1f}h | "
            markdown += f"{row['Cost per 1K']} |\n"
        
        return markdown
    
    def generate_insights(self, df: pd.DataFrame):
        """Generate key insights"""
        
        print("\n" + "="*100)
        print("KEY INSIGHTS FOR YOUR PAPER")
        print("="*100 + "\n")
        
        best_overall = df.iloc[0]
        best_free = df[df['Cost per 1K'] == '$0.00'].iloc[0] if len(df[df['Cost per 1K'] == '$0.00']) > 0 else None
        best_api = df[df['Type'] == 'LLM API'].iloc[0] if len(df[df['Type'] == 'LLM API']) > 0 else None
        
        print("1. BEST PERFORMANCE:")
        print(f"   {best_overall['Model']}: F1 = {best_overall['F1 Score']:.4f}")
        
        if best_free is not None:
            print(f"\n2. BEST FREE (LOCAL) MODEL:")
            print(f"   {best_free['Model']}: F1 = {best_free['F1 Score']:.4f}")
            print(f"   Training time: {best_free['Training Time (hrs)']:.1f} hours")
        
        if best_api is not None:
            print(f"\n3. BEST LLM API:")
            print(f"   {best_api['Model']}: F1 = {best_api['F1 Score']:.4f}")
            print(f"   Cost: {best_api['Cost per 1K']} per 1000 sentences")
        
        # Performance gap
        if best_free is not None and best_api is not None:
            gap = best_api['F1 Score'] - best_free['F1 Score']
            print(f"\n4. PERFORMANCE GAP:")
            print(f"   API advantage: {gap:.4f} F1 points ({gap*100:.2f}%)")
        
        # Cost analysis
        if best_api is not None:
            api_cost = float(best_api['Cost per 1K'].replace('$', ''))
            sentences_for_breakeven = 0  # Always free for local
            print(f"\n5. COST ANALYSIS:")
            print(f"   Local training: One-time {best_free['Training Time (hrs)']:.1f}h investment")
            print(f"   API cost: {best_api['Cost per 1K']} per 1K sentences")
            print(f"   For 100K sentences: ${api_cost * 100:.2f}")
        
        print("\n" + "="*100)
    
    def save_outputs(self, df: pd.DataFrame):
        """Save all outputs"""
        
        output_dir = Path("results/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV
        csv_file = output_dir / "comparison_table.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Saved CSV: {csv_file}")
        
        # LaTeX
        latex = self.create_latex_table(df)
        latex_file = output_dir / "comparison_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"✓ Saved LaTeX: {latex_file}")
        
        # Markdown
        markdown = self.create_markdown_table(df)
        md_file = output_dir / "comparison_table.md"
        with open(md_file, 'w') as f:
            f.write(markdown)
        print(f"✓ Saved Markdown: {md_file}")
        
        # JSON
        json_file = output_dir / "comparison_data.json"
        df.to_json(json_file, orient='records', indent=2)
        print(f"✓ Saved JSON: {json_file}")
    
    def run(self):
        """Run complete comparison"""
        
        print("\n" + "="*100)
        print("COMPARING ALL NER RESULTS")
        print("="*100)
        
        # Load results
        self.load_bert_results()
        self.load_llm_results()
        
        if not self.bert_results and not self.llm_results:
            print("\n No results found! Train models first:")
            print("  1. Fine-tuned: python train_models_sequentially.py")
            print("  2. LLM APIs: python evaluate_llm_apis.py --model gpt4 --api-key XXX")
            return
        df = self.create_comparison_table()
        
        self.print_comparison_table(df)
        
        self.generate_insights(df)
        
        self.save_outputs(df)
        
        print("\n" + "="*100)
        print("READY FOR YOUR PAPER!")
        print("="*100)
        print("\nUse the files in results/comparison/ for your paper:")
        print("  - comparison_table.csv  → Import to Excel/Google Sheets")
        print("  - comparison_table.tex  → Copy to LaTeX paper")
        print("  - comparison_table.md   → Use in GitHub/documentation")
        print("="*100 + "\n")


def main():
    comparator = ResultsComparator()
    comparator.run()


if __name__ == "__main__":
    main()

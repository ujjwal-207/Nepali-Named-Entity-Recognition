
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import defaultdict

class NERAPIEvaluator:
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        test_file: Path,
        output_dir: Path,
        num_shots: int = 0,
        max_samples: int = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.test_file = test_file
        self.output_dir = output_dir
        self.num_shots = num_shots
        self.max_samples = max_samples
        
        self.setup_api_client()
        
        self.total_cost = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.total_time = 0
        
    def setup_api_client(self):
        """Setup API client"""
        if self.model_name == "gpt4":
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model_id = "gpt-4-turbo-preview"
            self.cost_per_1k_input = 0.01
            self.cost_per_1k_output = 0.03
                
        elif self.model_name == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model_id = "claude-3-5-sonnet-20241022"
            self.cost_per_1k_input = 0.003
            self.cost_per_1k_output = 0.015
                
        elif self.model_name == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = "gemini-2.0-flash-exp"
            self.cost_per_1k_input = 0.0
            self.cost_per_1k_output = 0.0
                
        elif self.model_name.startswith("groq-"):
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            if self.model_name == "groq-llama": self.model_id = "llama-3.3-70b-versatile"
            elif self.model_name == "groq-mixtral": self.model_id = "mixtral-8x7b-32768"
            else: self.model_id = "llama-3.1-8b-instant"
            self.cost_per_1k_input = 0.0
            self.cost_per_1k_output = 0.0
        elif self.model_name == "mistral":
            try:
                from mistralai import Mistral
                self.client = Mistral(api_key=self.api_key)
                self.model_id = "mistral-large-latest" 
                self.cost_per_1k_input = 0.002 
                self.cost_per_1k_output = 0.006
            except ImportError:
                print("Install Mistral: pip install mistralai")
                raise
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def load_test_data(self) -> List[Tuple[str, List[Dict]]]:
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        text = " ".join(current_tokens)
                        entities = self._extract_entities(current_tokens, current_labels)
                        sentences.append((text, entities))
                        current_tokens, current_labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
        
        if current_tokens:
            sentences.append((" ".join(current_tokens), self._extract_entities(current_tokens, current_labels)))
        
        return sentences[:self.max_samples] if self.max_samples else sentences

    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        entities = []
        curr_text, curr_type = [], None
        for tok, lab in zip(tokens, labels):
            if lab.startswith('B-'):
                if curr_text: entities.append({"text": " ".join(curr_text), "type": curr_type})
                curr_type, curr_text = lab[2:], [tok]
            elif lab.startswith('I-') and curr_type:
                curr_text.append(tok)
            else:
                if curr_text: entities.append({"text": " ".join(curr_text), "type": curr_type})
                curr_text, curr_type = [], None
        if curr_text: entities.append({"text": " ".join(curr_text), "type": curr_type})
        return entities

    def create_prompt(self, text: str, examples: List[Tuple[str, List[Dict]]] = None) -> str:
        system = """You are an expert in Nepali Named Entity Recognition.
Extract: PERSON, LOCATION, ORGANIZATION, DATE.
Return ONLY a valid JSON array. Do not include explanations.
Format: [{"text": "entity", "type": "TYPE"}]"""
        
        prompt = system + "\n\n"
        if examples and self.num_shots > 0:
            for i, (ex_t, ex_e) in enumerate(examples[:self.num_shots], 1):
                prompt += f"Text: {ex_t}\nOutput: {json.dumps(ex_e, ensure_ascii=False)}\n\n"
        
        prompt += f"Text: {text}\nOutput:"
        return prompt

    def call_api(self, prompt: str) -> Tuple[str, int, int, float]:
        start_time = time.time()
        try:
            if self.model_name == "mistral":
                response = self.client.chat.complete(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = response.choices[0].message.content
                in_tok = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens
            
            elif self.model_name.startswith("groq-"):
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                output = response.choices[0].message.content
                in_tok = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens
            
            # (Keep your existing gpt4, claude, gemini blocks here...)
            elif self.model_name == "gpt4":
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                output = response.choices[0].message.content
                in_tok = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens
            else:
                output, in_tok, out_tok = "[]", 0, 0

            elapsed = time.time() - start_time
            self.total_tokens_input += in_tok
            self.total_tokens_output += out_tok
            self.total_time += elapsed
            self.total_cost += (in_tok/1000 * self.cost_per_1k_input) + (out_tok/1000 * self.cost_per_1k_output)
            
            return output, in_tok, out_tok, elapsed
            
        except Exception as e:
            print(f"API Error: {e}")
            return "[]", 0, 0, 0

    def parse_response(self, response: str) -> List[Dict]:
        try:
            match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(response.strip())
        except:
            return []

    def normalize_tag(self, tag: str) -> str:
        tag = tag.upper().strip()
        mapping = {
            "PER": "PERSON", "PRO": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "DAT": "DATE"
        }
        return mapping.get(tag, tag)

    def evaluate(self):
        test_data = self.load_test_data()
        examples = test_data[:self.num_shots] if self.num_shots > 0 else None
        eval_data = test_data[self.num_shots:]
        
        if self.max_samples: eval_data = eval_data[:self.max_samples]

        results, tp, fp, fn = [], 0, 0, 0
        
        for i, (text, true_entities) in enumerate(eval_data, 1):
            prompt = self.create_prompt(text, examples)
            response, in_tok, out_tok, elapsed = self.call_api(prompt)
            pred_entities = self.parse_response(response)
            
            true_set = {(e['text'].strip(), self.normalize_tag(e['type'])) for e in true_entities}
            pred_set = {(str(e.get('text','')).strip(), self.normalize_tag(str(e.get('type','')))) for e in pred_entities}
            
            cur_tp = len(true_set & pred_set)
            cur_fp = len(pred_set - true_set)
            cur_fn = len(true_set - pred_set)
            
            tp += cur_tp; fp += cur_fp; fn += cur_fn
            
            results.append({"text": text, "tp": cur_tp, "fp": cur_fp, "fn": cur_fn})
            if i % 10 == 0: print(f" Processed {i}/{len(eval_data)}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.print_summary(precision, recall, f1, len(eval_data))

    def print_summary(self, precision, recall, f1, num_samples):
        print(f"\nRESULTS FOR {self.model_name}")
        print(f"F1 Score: {f1*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}%")
        print(f"Total Cost: ${self.total_cost:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['gpt4', 'claude', 'gemini', 'groq-llama', 'groq-mixtral', 'mistral'])
    parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--test-file', type=str, default='data/raw/EverestNER-test-bio.txt')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='results/llm_api')
    
    args = parser.parse_args()
    evaluator = NERAPIEvaluator(args.model, args.api_key, Path(args.test_file), Path(args.output_dir), args.shots, args.max_samples)
    evaluator.evaluate()

if __name__ == "__main__":
    main()

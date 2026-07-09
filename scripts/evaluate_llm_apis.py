
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import defaultdict
import evaluate;
from typing import Optional
import random
from src.reproducibility import set_seed

class NERAPIEvaluator:
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        test_file: Path,
        output_dir: Path,
        num_shots: int = 0,
        max_samples: int = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.test_file = test_file
        self.output_dir = output_dir
        self.num_shots = num_shots
        self.max_samples = max_samples
        self.seed = seed
        set_seed(seed)
        
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
            if lab.startswith("B-"):
                if curr_text:
                    entities.append({"text": " ".join(curr_text), "type": curr_type})
                curr_type = lab[2:]
                curr_text = [tok]

            elif lab.startswith("I-"):
                lab_type = lab[2:]
                if curr_text and curr_type == lab_type:
                    curr_text.append(tok)
                else:
                # I- without correct preceding B- or type mismatch: close and restart
                    if curr_text:
                        entities.append({"text": " ".join(curr_text), "type": curr_type})
                    curr_type = lab_type
                    curr_text = [tok]
            else:
                if curr_text:
                    entities.append({"text": " ".join(curr_text), "type": curr_type})
                curr_text, curr_type = [], None

        if curr_text:
            entities.append({"text": " ".join(curr_text), "type": curr_type})

        return entities

    def create_prompt(self, text: str, examples: List[Tuple[str, List[Dict]]] = None) -> str:
        system = """You are doing Nepali Named Entity Recognition (EverestNER).Extract entities of these types ONLY: Person, Location, Organization, Date, Event.Return a JSON array ONLY (no markdown, no code fences, no extra text).Schema: [{"text":"...","type":"Person|Location|Organization|Date|Event"}]Do NOT return [] unless you are confident there are no entities in the text."""
        
        prompt = system + "\n\n"
        if examples and self.num_shots > 0:
            for i, (ex_t, ex_e) in enumerate(examples[:self.num_shots], 1):
                prompt += f"Text: {ex_t}\nOutput: {json.dumps(ex_e, ensure_ascii=False)}\n\n"
        
        prompt += f"Text: {text}\nOutput:"
        return prompt

    def call_api(self, prompt: str) -> Tuple[str, int, int, float,bool]:
        max_retries = 4
        base_sleep = 1.0
        for attempt in range(max_retries + 1):
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
            
                return output, in_tok, out_tok, elapsed,True
            
            except Exception as e:
                elapsed = time.time() - start_time
                msg = str(e)
                transient = ("Status 504" in msg) or ("429" in msg) or ("timing out" in msg) or ("5" in msg)
                if attempt < max_retries and transient:
                    sleep_s = base_sleep * (2 ** attempt) + random.random()
                    time.sleep(sleep_s)
                    continue
                print(f"API Error (final): {e}")
                return "[]", 0, 0,elapsed, False
            
    def clean_raw_for_json(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
        return s
    def parse_response(self, response: str) -> List[Dict]:
        s = self.clean_raw_for_json(response)
        if not s:
            return []

        
        if s == "[]":
            return []

    
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, list) else []
        except Exception:
            pass

    
        m = re.search(r"\[[\s\S]*?\]", s)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, list) else []
            except Exception:
                return []

        return []

    def normalize_tag(self, tag: str) -> str:
        tag = tag.upper().strip()
        mapping = {
            "PER": "PERSON", "PERSON": "PERSON", "PRO": "PERSON",
            "LOC": "LOCATION", "LOCATION": "LOCATION",
            "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION",
            "DAT": "DATE", "DATE": "DATE",
            "EVENT": "EVENT", "EVT": "EVENT",
        }
        return mapping.get(tag, tag)


    def load_conll_sentences(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
        sentences = []
        cur_tokens, cur_labels = [], []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if cur_tokens:
                        sentences.append((cur_tokens, cur_labels))
                        cur_tokens, cur_labels = [], []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    cur_tokens.append(parts[0])
                    cur_labels.append(parts[1])
        if cur_tokens:
            sentences.append((cur_tokens, cur_labels))
        if self.max_samples:
            rnd = random.Random(self.seed)
            rnd.shuffle(sentences)
            sentences = sentences[: self.max_samples]

        return sentences
    def detect_schema(self, train_sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, Optional[str]]:
        label_set = set()
        for _, labs in train_sentences:
            label_set.update(labs)

        types = set()
        for lab in label_set:
            if lab.startswith("B-") or lab.startswith("I-"):
                types.add(lab[2:])

        def pick(pref):
            for t in pref:
                if t in types:
                    return t
            return None

        return {
        "PERSON": pick(["PER", "Person", "PERSON"]),
        "LOCATION": pick(["LOC", "Location", "LOCATION", "GPE"]),
        "ORGANIZATION": pick(["ORG", "Organization", "ORGANIZATION"]),
        "DATE": pick(["DATE", "Date", "DAT"]),
        "EVENT": pick(["EVENT", "Event", "EVT"]),
        "MISC": pick(["MISC"]),
    }

    def entities_to_bio(self, tokens: List[str], pred_entities: List[Dict], schema: Dict[str, Optional[str]]) -> List[str]:
        pred_labels = ["O"] * len(tokens)
        used = [False] * len(tokens)

        # normalize entity list
        norm = []
        for e in pred_entities:
            text = str(e.get("text", "")).strip()
            typ = self.normalize_tag(str(e.get("type", "")))
            if not text or not typ:
                continue

            # Map canonical type -> dataset type
            ds_type = schema.get(typ)
            if not ds_type:
                # If DATE not in dataset but model outputs DATE, ignore it
                continue

            ent_toks = text.split()
            if ent_toks:
                norm.append((ent_toks, ds_type))

        # greedy: longer entities first
        norm.sort(key=lambda x: len(x[0]), reverse=True)

        def all_matches(ent_toks):
            n = len(ent_toks)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == ent_toks:
                    yield i

        for ent_toks, ds_type in norm:
            n = len(ent_toks)
            for start in all_matches(ent_toks):
                if any(used[start:start+n]):
                    continue
                pred_labels[start] = f"B-{ds_type}"
                for j in range(start + 1, start + n):
                    pred_labels[j] = f"I-{ds_type}"
                for j in range(start, start + n):
                    used[j] = True
                break

        return pred_labels

    def find_train_file(self) -> Path:
            candidates = [
            "data/raw/EverestNER-train-bio.txt",
            "data/raw/EverestNER-train-bio",
            "data/raw/train.bio",
            "data/raw/train.txt",
            ]
            for c in candidates:
                p = Path(c)
                if p.exists():
                    return p
            raise FileNotFoundError(f"Train file not found. Tried: {candidates}")


    def evaluate(self):
        # 1) Load TRAIN (for few-shot examples) and TEST (for evaluation)
        api_fail = 0
        train_file = self.find_train_file()
        train_sents = self.load_conll_sentences(train_file)

        test_file = Path(self.test_file)
        test_sents = self.load_conll_sentences(test_file)

        # 2) Detect dataset label schema (PER/LOC/ORG vs PERSON/LOCATION/ORG etc.)
        schema = self.detect_schema(train_sents)

        # 3) Prepare few-shot examples from TRAIN (NOT TEST)
        examples = None
        if self.num_shots > 0:
            examples = []
            # use first N train examples (deterministic)
            for toks, labs in train_sents[: self.num_shots]:
                ex_text = " ".join(toks)
                ex_entities = self._extract_entities(toks, labs)

                # Convert dataset tags (PER/LOC/ORG) -> canonical (PERSON/LOCATION/ORGANIZATION/DATE)
                reverse = {v: k for k, v in schema.items() if v}
                canon_entities = []
                for ent in ex_entities:
                    canon_entities.append({
                        "text": ent["text"],
                        "type": reverse.get(ent["type"], ent["type"])
                    })

                examples.append((ex_text, canon_entities))

        # 4) Evaluate using seqeval (same metric as your encoder baselines)
        metric = evaluate.load("seqeval")
        y_true, y_pred = [], []
        parse_fail = 0

        # Logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"{self.model_name}_shots{self.num_shots}_n{len(test_sents)}_seed{self.seed}.jsonl"


        with open(log_path, "w", encoding="utf-8") as logf:
            for i, (toks, gold_labs) in enumerate(test_sents, 1):
                text = " ".join(toks)
                prompt = self.create_prompt(text, examples)

                raw, in_tok, out_tok, elapsed,ok = self.call_api(prompt)
                if not ok:
                    api_fail += 1;
                pred_entities = self.parse_response(raw)

                # track parse failures (non-empty output that parsed to [])
                cleaned = self.clean_raw_for_json(raw)
                if cleaned not in ("", "[]") and pred_entities == []:
                    parse_fail += 1

                pred_labs = self.entities_to_bio(toks, pred_entities, schema)

                y_true.append(gold_labs)
                y_pred.append(pred_labs)

                logf.write(json.dumps({
                    "idx": i,
                    "text": text,
                    "tokens": toks,
                    "gold": gold_labs,
                    "pred": pred_labs,
                    "raw": raw,
                    "parsed": pred_entities,
                    "latency_s": elapsed,
                    "in_tok": in_tok,
                    "out_tok": out_tok
                }, ensure_ascii=False) + "\n")

                if i % 10 == 0:
                    print(f"Processed {i}/{len(test_sents)}")

        results = metric.compute(predictions=y_pred, references=y_true)
        precision = results.get("overall_precision", 0.0)
        wrecall = results.get("overall_recall", 0.0)
        f1 = results.get("overall_f1", 0.0)
        print(f"API failures: {api_fail}/{len(test_sents)} ({api_fail/len(test_sents)*100:.1f}%)")
        print(f"\nRESULTS FOR {self.model_name} (seqeval entity-level)")
        print(f"F1: {f1*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {wrecall*100:.2f}%")
        
        # Per-entity breakdown
        print(f"\n  Per-Entity Breakdown:")
        for entity in ["Person", "Location", "Organization", "Date", "Event"]:
            if entity in results:
                e = results[entity]
                print(f"    {entity:<15} P:{float(e['precision']):.3f}  R:{float(e['recall']):.3f}  F1:{float(e['f1']):.3f}  N:{int(e['number'])}")
        print(f"Parse failures: {parse_fail}/{len(test_sents)} ({(parse_fail/len(test_sents))*100:.1f}%)")
        print(f"Avg latency: {(self.total_time/len(test_sents)):.2f}s/sample")
        print(f"Total cost: ${self.total_cost:.4f}")
        # Save per-entity to audit log
        per_entity = {}
        for entity in ["Person", "Location", "Organization", "Date", "Event"]:
            if entity in results:
                per_entity[entity] = {
                    "precision": float(results[entity]["precision"]),
                    "recall": float(results[entity]["recall"]),
                    "f1": float(results[entity]["f1"]),
                    "support": int(results[entity]["number"])
                }
        
        # Write summary at end of audit log
        summary = {"overall": {"precision": float(precision), "recall": float(wrecall), "f1": float(f1)}, "per_entity": per_entity}
        logf.write(json.dumps({"__summary__": summary}, ensure_ascii=False) + "\n")
        
        print(f"Saved audit log: {log_path}")

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
    parser.add_argument('--seed', type=int, default=42)

    
    args = parser.parse_args()
    evaluator = NERAPIEvaluator(args.model, args.api_key, Path(args.test_file), Path(args.output_dir), args.shots, args.max_samples,seed=args.seed)
    evaluator.evaluate()

if __name__ == "__main__":
    main()

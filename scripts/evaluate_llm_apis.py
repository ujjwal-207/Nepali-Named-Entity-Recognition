import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import os
from collections import defaultdict
import evaluate
import numpy as np
import random
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reproducibility import set_seed


class NERAPIEvaluator:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        test_file: Path = Path("data/raw/EverestNER-test-bio.txt"),
        output_dir: Path = Path("results/llm_api"),
        num_shots: int = 0,
        max_samples: Optional[int] = None,
        model_id: Optional[str] = None,
        seed: int = 42,
        delay: float = 0.1,
        resume: bool = True,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.test_file = Path(test_file)
        self.output_dir = Path(output_dir)
        self.num_shots = num_shots
        self.max_samples = max_samples
        self.model_id_override = model_id
        self.seed = seed
        self.delay = delay
        self.resume = resume
        set_seed(seed)

        self.total_cost = 0.0
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.total_time = 0.0

        self.setup_api_client()

    def setup_api_client(self):
        """Setup unified API client for OpenRouter."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenRouter evaluation requires the openai package. Run: pip install openai")

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/ujjwal-207/Nepali-Named-Entity-Recognition",
                "X-Title": "Nepali NER Benchmark",
            }
        )
        self.provider = "openrouter"

        # Resolve model slug
        name = self.model_name.lower()
        if self.model_id_override:
            self.model_id = self.model_id_override
            if self.model_id in ["openai/gpt-4-mini", "gpt-4-mini", "gpt4-mini"]:
                self.model_id = "openai/gpt-4o-mini"
        elif "gpt" in name:
            self.model_id = "openai/gpt-4o-mini"
        elif "llama-small" in name or "8b" in name:
            self.model_id = "meta-llama/llama-3.1-8b-instruct"
        elif "llama" in name:
            self.model_id = "meta-llama/llama-3.3-70b-instruct"
        elif "qwen" in name:
            self.model_id = "qwen/qwen-2.5-72b-instruct"
        else:
            self.model_id = self.model_name

        # OpenRouter pricing per 1K tokens
        if "llama-3.3-70b" in self.model_id:
            self.cost_per_1k_input = 0.00012
            self.cost_per_1k_output = 0.00030
        elif "qwen-2.5-72b" in self.model_id:
            self.cost_per_1k_input = 0.00035
            self.cost_per_1k_output = 0.00040
        elif "8b" in self.model_id:
            self.cost_per_1k_input = 0.00005
            self.cost_per_1k_output = 0.00005
        elif "gpt-4o-mini" in self.model_id:
            self.cost_per_1k_input = 0.00015
            self.cost_per_1k_output = 0.00060
        else:
            self.cost_per_1k_input = 0.0005
            self.cost_per_1k_output = 0.0010

        print(f"Initialized OpenRouter with Model ID: {self.model_id}")

    def load_conll_sentences(self, file_path: Path, limit: Optional[int] = None) -> List[Tuple[List[str], List[str]]]:
        """Loads CoNLL BIO sentences sequentially without corrupting original order."""
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

        if limit is not None:
            sentences = sentences[:limit]

        return sentences

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

    def create_prompt(self, text: str, examples: Optional[List[Tuple[str, List[Dict]]]] = None) -> str:
        system = (
            "You are doing Nepali Named Entity Recognition (EverestNER).\n"
            "Extract entities of these types ONLY: Person, Location, Organization, Date, Event.\n"
            "Return a JSON array ONLY (no markdown code blocks, no explanations, no extra text).\n"
            'Schema: [{"text": "...", "type": "Person|Location|Organization|Date|Event"}]\n'
            "If no entities are present, return []."
        )

        prompt = system + "\n\n"
        if examples and self.num_shots > 0:
            for ex_t, ex_e in examples[: self.num_shots]:
                prompt += f"Text: {ex_t}\nOutput: {json.dumps(ex_e, ensure_ascii=False)}\n\n"

        prompt += f"Text: {text}\nOutput:"
        return prompt

    def call_api(self, prompt: str) -> Tuple[str, int, int, float, bool]:
        max_retries = 6
        base_sleep = 2.0

        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=250,
                )
                output = response.choices[0].message.content or "[]"
                in_tok = response.usage.prompt_tokens if response.usage else len(prompt.split()) * 2
                out_tok = response.usage.completion_tokens if response.usage else len(output.split()) * 2

                elapsed = time.time() - start_time
                self.total_tokens_input += in_tok
                self.total_tokens_output += out_tok
                self.total_time += elapsed
                self.total_cost += (in_tok / 1000 * self.cost_per_1k_input) + (out_tok / 1000 * self.cost_per_1k_output)

                return output, in_tok, out_tok, elapsed, True

            except Exception as e:
                elapsed = time.time() - start_time
                msg = str(e)
                is_transient = any(err in msg for err in ["429", "504", "503", "500", "502", "rate_limit", "timing out", "Connection"])

                if attempt < max_retries and is_transient:
                    sleep_s = min(60.0, base_sleep * (2 ** attempt) + random.uniform(0.5, 2.0))
                    print(f"⚠️ [Attempt {attempt+1}/{max_retries}] Transient error: {msg[:100]}... Sleeping {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue

                print(f"❌ API Error (final attempt failed): {e}")
                return "[]", 0, 0, elapsed, False

    def clean_raw_for_json(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
        return s

    def parse_response(self, response: str) -> List[Dict]:
        s = self.clean_raw_for_json(response)
        if not s or s == "[]":
            return []

        # 1. Direct JSON parse
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

        # 2. Extract first valid JSON array via regex (handles trailing commentary/hallucination)
        match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", s)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass

        # 3. Fallback for empty brackets
        if re.search(r"\[\s*\]", s):
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

    def entities_to_bio(self, tokens: List[str], pred_entities: List[Dict], schema: Dict[str, Optional[str]]) -> List[str]:
        pred_labels = ["O"] * len(tokens)
        used = [False] * len(tokens)

        norm = []
        for e in pred_entities:
            text = str(e.get("text", "")).strip()
            typ = self.normalize_tag(str(e.get("type", "")))
            if not text or not typ:
                continue

            ds_type = schema.get(typ)
            if not ds_type:
                continue

            ent_toks = text.split()
            if ent_toks:
                norm.append((ent_toks, ds_type))

        # Greedy: match longer entities first
        norm.sort(key=lambda x: len(x[0]), reverse=True)

        def all_matches(ent_toks):
            n = len(ent_toks)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == ent_toks:
                    yield i

        for ent_toks, ds_type in norm:
            n = len(ent_toks)
            for start in all_matches(ent_toks):
                if any(used[start:start + n]):
                    continue
                pred_labels[start] = f"B-{ds_type}"
                for j in range(start + 1, start + n):
                    pred_labels[j] = f"I-{ds_type}"
                for j in range(start, start + n):
                    used[j] = True
                break

        return pred_labels

    def detect_schema(self, train_sents: List[Tuple[List[str], List[str]]]) -> Dict[str, str]:
        tags = set()
        for _, labs in train_sents:
            for l in labs:
                if l != "O" and "-" in l:
                    tags.add(l.split("-", 1)[1])

        def pick(options):
            for opt in options:
                if opt in tags:
                    return opt
            return None

        return {
            "PERSON": pick(["PER", "Person", "PERSON"]),
            "LOCATION": pick(["LOC", "Location", "LOCATION"]),
            "ORGANIZATION": pick(["ORG", "Organization", "ORGANIZATION"]),
            "DATE": pick(["DATE", "Date", "DAT"]),
            "EVENT": pick(["EVENT", "Event", "EVT"]),
        }

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

    def compute_bootstrap_ci(self, y_true, y_pred, n_iterations: int = 500) -> Tuple[float, float, float]:
        """Calculates 95% Confidence Interval via 500-iteration bootstrapping."""
        metric = evaluate.load("seqeval")
        bootstrapped_f1s = []
        n_samples = len(y_true)
        indices = np.arange(n_samples)

        print(f"  Calculating 95% Bootstrap CI ({n_iterations} iterations)...")
        for _ in range(n_iterations):
            batch_indices = np.random.choice(indices, size=n_samples, replace=True)
            sample_true = [y_true[i] for i in batch_indices]
            sample_pred = [y_pred[i] for i in batch_indices]

            res = metric.compute(predictions=sample_pred, references=sample_true)
            bootstrapped_f1s.append(res.get("overall_f1", 0.0))

        bootstrapped_f1s.sort()
        low = bootstrapped_f1s[int(0.025 * n_iterations)]
        high = bootstrapped_f1s[int(0.975 * n_iterations)]
        margin = (high - low) / 2
        return low, high, margin

    def evaluate(self):
        train_file = self.find_train_file()
        train_sents = self.load_conll_sentences(train_file)
        test_sents = self.load_conll_sentences(self.test_file, limit=self.max_samples)

        schema = self.detect_schema(train_sents)
        print(f"Detected Label Schema: {schema}")

        # Build few-shot examples from train set
        examples = []
        if self.num_shots > 0:
            for toks, labs in train_sents[: self.num_shots]:
                ex_text = " ".join(toks)
                ex_entities = self._extract_entities(toks, labs)
                reverse = {v: k for k, v in schema.items() if v}
                canon_entities = [
                    {"text": ent["text"], "type": reverse.get(ent["type"], ent["type"])}
                    for ent in ex_entities
                ]
                examples.append((ex_text, canon_entities))

        slug_source = self.model_id if hasattr(self, "model_id") and self.model_id else self.model_name
        safe_model_slug = slug_source.replace("/", "_").replace(":", "_")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"{safe_model_slug}_shots{self.num_shots}_n{len(test_sents)}_seed{self.seed}.jsonl"

        # Resume support: Read existing completed entries
        completed_records = {}
        if self.resume and log_path.exists():
            print(f"🔄 Checkpoint detected at {log_path}. Reading completed records...")
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "__summary__" in record:
                            continue
                        idx = record.get("idx")
                        if idx is not None:
                            completed_records[idx] = record
                    except Exception:
                        pass
            print(f"✅ Found {len(completed_records)} already completed sentences. Resuming from sample {len(completed_records)+1}...")

        y_true, y_pred = [], []
        parse_fail = 0
        api_fail = 0

        # Open in append mode so prior progress is never lost
        mode = "a" if (self.resume and log_path.exists()) else "w"
        with open(log_path, mode, encoding="utf-8") as logf:
            for i, (toks, gold_labs) in enumerate(test_sents, 1):
                # If already completed in previous run, reload from log
                if i in completed_records:
                    rec = completed_records[i]
                    y_true.append(rec["gold"])
                    y_pred.append(rec["pred"])
                    self.total_tokens_input += rec.get("in_tok", 0)
                    self.total_tokens_output += rec.get("out_tok", 0)
                    self.total_time += rec.get("latency_s", 0.0)
                    continue

                text = " ".join(toks)
                prompt = self.create_prompt(text, examples)

                raw, in_tok, out_tok, elapsed, ok = self.call_api(prompt)
                if not ok:
                    api_fail += 1

                pred_entities = self.parse_response(raw)

                cleaned = self.clean_raw_for_json(raw)
                if cleaned not in ("", "[]") and pred_entities == []:
                    parse_fail += 1

                pred_labs = self.entities_to_bio(toks, pred_entities, schema)
                y_true.append(gold_labs)
                y_pred.append(pred_labs)

                record = {
                    "idx": i,
                    "text": text,
                    "tokens": toks,
                    "gold": gold_labs,
                    "pred": pred_labs,
                    "raw": raw,
                    "parsed": pred_entities,
                    "latency_s": elapsed,
                    "in_tok": in_tok,
                    "out_tok": out_tok,
                }
                logf.write(json.dumps(record, ensure_ascii=False) + "\n")
                logf.flush()

                if i % 10 == 0 or i == len(test_sents):
                    print(f"Progress: [{i}/{len(test_sents)}] | Latency: {elapsed:.2f}s | Est. Cost: ${self.total_cost:.4f}")

                if self.delay > 0:
                    time.sleep(self.delay)

        # Final Metrics Computation
        metric = evaluate.load("seqeval")
        results = metric.compute(predictions=y_pred, references=y_true)
        precision = results.get("overall_precision", 0.0)
        recall = results.get("overall_recall", 0.0)
        f1 = results.get("overall_f1", 0.0)

        f1_low, f1_high, f1_margin = self.compute_bootstrap_ci(y_true, y_pred, n_iterations=500)

        print("\n" + "=" * 65)
        print(f"EVALUATION RESULTS FOR {self.model_name} ({self.num_shots}-shot)")
        print(f"   F1-Score:    {f1*100:.2f}% (±{f1_margin*100:.2f}%)")
        print(f"   95% CI:      [{f1_low*100:.2f}%, {f1_high*100:.2f}%]")
        print(f"   Precision:   {precision*100:.2f}%")
        print(f"   Recall:      {recall*100:.2f}%")
        print(f"   Total Time:  {self.total_time/60:.2f} minutes (Avg: {self.total_time/len(test_sents):.2f}s/sent)")
        print(f"   Total Cost:  ${self.total_cost:.4f}")
        print(f"   Parse Fails: {parse_fail}/{len(test_sents)} ({parse_fail/len(test_sents)*100:.1f}%)")
        print(f"   API Fails:   {api_fail}/{len(test_sents)} ({api_fail/len(test_sents)*100:.1f}%)")
        print("=" * 65)

        # Per-Entity Breakdown
        per_entity = {}
        print("\n  Per-Entity Breakdown:")
        for entity in ["Person", "Location", "Organization", "Date", "Event"]:
            if entity in results:
                e = results[entity]
                per_entity[entity] = {
                    "precision": float(e["precision"]),
                    "recall": float(e["recall"]),
                    "f1": float(e["f1"]),
                    "support": int(e["number"]),
                }
                print(f"    {entity:<15} P: {float(e['precision'])*100:5.2f}%  R: {float(e['recall'])*100:5.2f}%  F1: {float(e['f1'])*100:5.2f}%  Support: {int(e['number'])}")

        summary = {
            "model": self.model_name,
            "model_id": self.model_id,
            "shots": self.num_shots,
            "num_samples": len(test_sents),
            "overall": {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "f1_ci_low": float(f1_low),
                "f1_ci_high": float(f1_high),
                "f1_margin": float(f1_margin),
            },
            "per_entity": per_entity,
            "computational": {
                "total_time_seconds": self.total_time,
                "avg_latency_s": self.total_time / len(test_sents),
                "total_cost_usd": self.total_cost,
                "input_tokens": self.total_tokens_input,
                "output_tokens": self.total_tokens_output,
            },
            "failures": {
                "parse_failures": parse_fail,
                "api_failures": api_fail,
            }
        }

        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(json.dumps({"__summary__": summary}, ensure_ascii=False) + "\n")

        print(f"\nAudit log and summary saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM APIs for Nepali NER with Checkpointing & OpenRouter Support")
    parser.add_argument("--model", type=str, default="openrouter",
                        help="Model family or alias (openrouter, llama, qwen, gpt, llama-small)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Exact model slug (e.g., 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-2.5-72b-instruct')")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (optional if OPENROUTER_API_KEY environment variable is set)")
    parser.add_argument("--test-file", type=str, default="data/raw/EverestNER-test-bio.txt")
    parser.add_argument("--shots", type=int, default=0, help="Number of in-context few-shot examples (0 or 5)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples (default: None = all 2,078)")
    parser.add_argument("--output-dir", type=str, default="results/llm_api")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls in seconds (default: 0.1s)")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing logs instead of resuming")

    args = parser.parse_args()

    evaluator = NERAPIEvaluator(
        model_name=args.model,
        api_key=args.api_key,
        test_file=Path(args.test_file),
        output_dir=Path(args.output_dir),
        num_shots=args.shots,
        max_samples=args.max_samples,
        model_id=args.model_id,
        seed=args.seed,
        delay=args.delay,
        resume=not args.no_resume,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()

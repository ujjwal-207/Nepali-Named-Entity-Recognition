from pathlib import Path
import json
from typing import List, Dict, Tuple
from collections import defaultdict


class BioToInstructionConverter:
    
    def __init__(self, bio_file: Path, output_file: Path, template: str = "llama"):
        self.bio_file = bio_file
        self.output_file = output_file
        self.template = template
        self.entity_stats = defaultdict(int)
        
    def load_bio_data(self) -> List[Tuple[List[str], List[str]]]:
        sentences = []
        labels = []
        current_tokens = []
        current_labels = []
        
        print(f"\n Loading: {self.bio_file.name}")
        
        with open(self.bio_file, 'r', encoding='utf-8') as f:
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
        
        print(f" Loaded {len(sentences)} sentences")
        return list(zip(sentences, labels))
    
    def extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        entities = []
        current_entity = None
        current_text = []
        current_type = None
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current_entity is not None:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_type,
                        "label": current_type  
                    })
                    self.entity_stats[current_type] += 1
                
                current_type = label[2:]
                current_text = [token]
                current_entity = True
                
            elif label.startswith('I-') and current_entity:
                current_text.append(token)
                
            else: 
                if current_entity is not None:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_type,
                        "label": current_type
                    })
                    self.entity_stats[current_type] += 1
                    current_entity = None
                    current_text = []
                    current_type = None
        
        if current_entity is not None:
            entities.append({
                "text": " ".join(current_text),
                "type": current_type,
                "label": current_type
            })
            self.entity_stats[current_type] += 1
        
        return entities
    
    def format_llama_style(self, text: str, entities: List[Dict]) -> Dict:
        
        system = "You are an expert in Named Entity Recognition for Nepali text. Extract all named entities and classify them into categories like PERSON, LOCATION, ORGANIZATION, DATE, TIME, etc."
        
        instruction = f"""Extract all named entities from the following Nepali text:

Text: {text}

Provide the entities in JSON format with 'text' and 'type' fields."""
        
        if entities:
            response = json.dumps(entities, ensure_ascii=False, indent=2)
        else:
            response = "[]"
        
        return {
            "system": system,
            "instruction": instruction,
            "input": text,
            "output": response,
            "entities": entities,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ],
            "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>",
        }
    
    def format_simple_style(self, text: str, entities: List[Dict]) -> Dict:
        
        instruction = f"### Instruction:\nExtract named entities from the Nepali text below.\n\n### Input:\n{text}\n\n### Output:"
        
        if entities:
            output = json.dumps(entities, ensure_ascii=False)
        else:
            output = "[]"
        
        return {
            "instruction": instruction,
            "input": text,
            "output": output,
            "entities": entities,
            "text": f"{instruction}\n{output}"
        }
    
    def format_chat_style(self, text: str, entities: List[Dict]) -> Dict:
        
        user_msg = f"Extract all named entities from this Nepali text and return them as JSON:\n\n{text}"
        
        if entities:
            assistant_msg = json.dumps(entities, ensure_ascii=False, indent=2)
        else:
            assistant_msg = "[]"
        
        return {
            "input": text,
            "entities": entities,
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "text": f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        }
    
    def convert(self):
        
        
        data = self.load_bio_data()
        
        converted_data = []
        
        for tokens, labels in data:
            text = " ".join(tokens)
            
            entities = self.extract_entities(tokens, labels)
            
            if self.template == "llama":
                formatted = self.format_llama_style(text, entities)
            elif self.template == "simple":
                formatted = self.format_simple_style(text, entities)
            elif self.template == "chat":
                formatted = self.format_chat_style(text, entities)
            else:
                raise ValueError(f"Unknown template: {self.template}")
            
            converted_data.append(formatted)
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        
        total_entities = sum(self.entity_stats.values())
        for entity_type, count in sorted(self.entity_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            print(f"  {entity_type:<20} {count:>6} ({percentage:>5.1f}%)")
        print(f"  {'TOTAL':<20} {total_entities:>6}")
        print(f"{'='*70}\n")
    
    def show_examples(self, num_examples: int = 3):
        
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break
                
                data = json.loads(line)
                
                print(f"Example {i+1}:")
                print(f"{'-'*70}")
                print(f"Original text: {data.get('input', 'N/A')[:100]}...")
                print(f"\nEntities found: {len(data.get('entities', []))}")
                
                if data.get('entities'):
                    print(f"\nEntities:")
                    for entity in data['entities'][:5]:  # Show first 5
                        print(f"  - {entity['text']:<30} [{entity['type']}]")
                
                if self.template == "llama" and 'messages' in data:
                    print(f"\nLlama Format Preview:")
                    print(f"System: {data['messages'][0]['content'][:80]}...")
                    print(f"User:   {data['messages'][1]['content'][:80]}...")
                    print(f"Output: {data['messages'][2]['content'][:80]}...")
                
                print(f"{'='*70}\n")


def main():
    
    
    data_dir = Path("data/raw")
    output_dir = Path("data/instruction_format")
    
    possible_files = [
        "EverestNER-train-bio",
        "EverestNER-train-bio.txt",
        "train.bio",
        "train.txt"
    ]
    
    train_file = None
    for filename in possible_files:
        test_path = data_dir / filename
        if test_path.exists():
            train_file = test_path
            break
    
    if not train_file:
        print(f"\n  file not found in: {data_dir}")
        print(f"Tried: {', '.join(possible_files)}")
        return
    
    print("  1. llama  - Llama 3.2 format (with system prompt)")
    print("  2. chat   - Simple chat format (Mistral, Phi)")
    print("  3. simple - Basic instruction format (any model)")
    
    choice = input("\nChoose template (1/2/3) or press Enter for llama: ").strip()
    
    template_map = {"1": "llama", "2": "chat", "3": "simple", "": "llama"}
    template = template_map.get(choice, "llama")
    
    
    converter = BioToInstructionConverter(
        bio_file=train_file,
        output_file=output_dir / f"train_{template}.jsonl",
        template=template
    )
    converter.convert()
    converter.show_examples(num_examples=3)
    
    val_file = data_dir / "EverestNER-val-bio"
    if not val_file.exists():
        val_file = data_dir / "EverestNER-val-bio.txt"
    
    if val_file.exists():
        
        val_converter = BioToInstructionConverter(
            bio_file=val_file,
            output_file=output_dir / f"val_{template}.jsonl",
            template=template
        )
        val_converter.convert()
    
    test_file = data_dir / "EverestNER-test-bio"
    if not test_file.exists():
        test_file = data_dir / "EverestNER-test-bio.txt"
    
    if test_file.exists():
        
        test_converter = BioToInstructionConverter(
            bio_file=test_file,
            output_file=output_dir / f"test_{template}.jsonl",
            template=template
        )
        test_converter.convert()
    
    print(f"  - train_{template}.jsonl")
    if val_file.exists():
        print(f"  - val_{template}.jsonl")
    if test_file.exists():
        print(f"  - test_{template}.jsonl")
    


if __name__ == "__main__":
    main()

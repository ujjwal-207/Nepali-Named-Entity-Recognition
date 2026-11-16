"""
Utility to convert CoNLL BIO annotations into instruction-tuning datasets.

Supports multiple conversational formats (Llama, ChatML, and Alpaca/Simple).
"""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class BioToInstructionConverter:
    """Converts BIO annotated token files into instruction-formatted JSON lines."""

    def __init__(
        self, bio_file: Path, output_file: Path, template: str = "llama"
    ) -> None:
        """Initialize the instruction dataset converter.

        Args:
            bio_file: Input CoNLL BIO file path.
            output_file: Output JSONL destination file path.
            template: Prompt template format ('llama', 'chat', or 'simple').
        """
        self.bio_file = Path(bio_file)
        self.output_file = Path(output_file)
        self.template = template
        self.entity_stats: Dict[str, int] = defaultdict(int)

    def load_bio_data(self) -> List[Tuple[List[str], List[str]]]:
        """Parse the input BIO file into token and label lists."""
        sentences: List[List[str]] = []
        labels: List[List[str]] = []
        current_tokens: List[str] = []
        current_labels: List[str] = []

        logger.info("Loading BIO annotations from %s", self.bio_file)

        with open(self.bio_file, "r", encoding="utf-8") as f:
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

        logger.info("Loaded %d sentences", len(sentences))
        return list(zip(sentences, labels))

    def extract_entities(
        self, tokens: List[str], labels: List[str]
    ) -> List[Dict[str, str]]:
        """Extract continuous entity spans from BIO tags."""
        entities: List[Dict[str, str]] = []
        current_text: List[str] = []
        current_type: str = ""
        in_entity: bool = False

        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if in_entity:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_type,
                    })
                    self.entity_stats[current_type] += 1
                current_type = label[2:]
                current_text = [token]
                in_entity = True
            elif label.startswith("I-") and in_entity:
                current_text.append(token)
            else:
                if in_entity:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_type,
                    })
                    self.entity_stats[current_type] += 1
                    in_entity = False
                    current_text = []
                    current_type = ""

        if in_entity:
            entities.append({
                "text": " ".join(current_text),
                "type": current_type,
            })
            self.entity_stats[current_type] += 1

        return entities

    def format_llama_style(
        self, text: str, entities: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Format an instance following the Llama 3 prompt structure."""
        system = (
            "You are an expert in Named Entity Recognition for Nepali text. "
            "Extract all named entities and classify them into categories: "
            "PERSON, LOCATION, ORGANIZATION, DATE, and EVENT."
        )
        instruction = (
            f"Extract all named entities from the following Nepali text:\n\n"
            f"Text: {text}\n\n"
            f"Provide the entities in JSON format with 'text' and 'type' fields."
        )
        response = json.dumps(entities, ensure_ascii=False) if entities else "[]"

        return {
            "system": system,
            "instruction": instruction,
            "input": text,
            "output": response,
            "entities": entities,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
        }

    def format_simple_style(
        self, text: str, entities: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Format an instance using standard instruction/input/output format."""
        instruction = "Extract named entities from the Nepali text below."
        response = json.dumps(entities, ensure_ascii=False) if entities else "[]"

        return {
            "instruction": instruction,
            "input": text,
            "output": response,
            "entities": entities,
        }

    def convert(self) -> None:
        """Run the dataset conversion and save to JSON Lines."""
        data = self.load_bio_data()
        converted_data: List[Dict[str, Any]] = []

        for tokens, labels in data:
            text = " ".join(tokens)
            entities = self.extract_entities(tokens, labels)

            if self.template == "llama":
                formatted = self.format_llama_style(text, entities)
            elif self.template == "simple":
                formatted = self.format_simple_style(text, entities)
            else:
                formatted = self.format_simple_style(text, entities)

            converted_data.append(formatted)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        total = sum(self.entity_stats.values())
        logger.info(
            "Conversion complete. Generated %d samples with %d total entities.",
            len(converted_data),
            total,
        )
        for ent_type, count in sorted(
            self.entity_stats.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info("  %-15s: %d", ent_type, count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CoNLL BIO to instruction tuning format."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/raw/EverestNER-train-bio.txt",
        help="Path to source CoNLL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/train_instruction.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="llama",
        choices=["llama", "simple"],
        help="Prompt formatting template.",
    )
    args = parser.parse_args()

    converter = BioToInstructionConverter(
        bio_file=Path(args.input_file),
        output_file=Path(args.output_file),
        template=args.template,
    )
    converter.convert()


if __name__ == "__main__":
    main()

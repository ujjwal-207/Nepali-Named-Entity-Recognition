"""
CoNLL BIO to text sequence conversion utility.

Provides formatting functions to convert token-tag sequences into bracketed,
XML, or inline tagged representations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class BIOConverter:
    """Converts token-level BIO tags into structured inline text formats."""

    def __init__(self, delimiter_style: str = "bracket") -> None:
        """Initialize the converter.

        Args:
            delimiter_style: One of 'bracket' ([text|TYPE]), 'xml' (<TYPE>text</TYPE>),
                or 'asterisk' (**text**(TYPE)).
        """
        self.delimiter_style = delimiter_style

    def parse_bio_tokens(self, tokens: List[str], tags: List[str]) -> str:
        """Parse aligned token and tag sequences into formatted inline text.

        Args:
            tokens: List of word tokens.
            tags: Aligned BIO tags.

        Returns:
            Formatted string representation.
        """
        result: List[str] = []
        current_entity: List[str] = []
        current_type: Optional[str] = None

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current_entity and current_type:
                    result.append(self._format_entity(current_entity, current_type))
                    current_entity = []
                current_type = tag[2:]
                current_entity = [token]
            elif tag.startswith("I-"):
                if current_entity and current_type:
                    current_entity.append(token)
                else:
                    current_type = tag[2:]
                    current_entity = [token]
            else:
                if current_entity and current_type:
                    result.append(self._format_entity(current_entity, current_type))
                    current_entity = []
                    current_type = None
                result.append(token)

        if current_entity and current_type:
            result.append(self._format_entity(current_entity, current_type))

        return " ".join(result)

    def _format_entity(self, entity_tokens: List[str], entity_type: str) -> str:
        """Format an entity span according to delimiter style."""
        entity_text = " ".join(entity_tokens)
        if self.delimiter_style == "bracket":
            return f"[{entity_text}|{entity_type}]"
        elif self.delimiter_style == "xml":
            return f"<{entity_type}>{entity_text}</{entity_type}>"
        elif self.delimiter_style == "asterisk":
            return f"**{entity_text}**({entity_type})"
        return f"[{entity_text}|{entity_type}]"

    def convert_from_conll(
        self, input_file: Path, output_file: Path
    ) -> List[Dict[str, Any]]:
        """Read a CoNLL file and output formatted JSON dataset."""
        sentences: List[Dict[str, Any]] = []
        current_tokens: List[str] = []
        current_tags: List[str] = []

        logger.info("Reading CoNLL file: %s", input_file)

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        formatted = self.parse_bio_tokens(current_tokens, current_tags)
                        sentences.append({
                            "original_tokens": current_tokens.copy(),
                            "bio_tags": current_tags.copy(),
                            "formatted_text": formatted,
                        })
                        current_tokens = []
                        current_tags = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_tags.append(parts[1])
                    elif len(parts) == 1:
                        current_tokens.append(parts[0])
                        current_tags.append("O")

        if current_tokens:
            formatted = self.parse_bio_tokens(current_tokens, current_tags)
            sentences.append({
                "original_tokens": current_tokens,
                "bio_tags": current_tags,
                "formatted_text": formatted,
            })

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)

        logger.info("Converted %d sentences. Saved to %s", len(sentences), output_file)
        return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CoNLL BIO data to structured JSON.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/raw/EverestNER-train-bio.txt",
        help="Input CoNLL file path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Target output directory.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="bracket",
        choices=["bracket", "xml", "asterisk"],
        help="Delimiter style for entity tags.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return

    converter = BIOConverter(delimiter_style=args.style)
    output_file = Path(args.output_dir) / f"formatted_{input_path.stem}_{args.style}.json"
    converter.convert_from_conll(input_path, output_file)


if __name__ == "__main__":
    main()

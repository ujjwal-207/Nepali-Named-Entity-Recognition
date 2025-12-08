
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


class BIOConverter:
    """Convert BIO-tagged data to LLM prompt format"""
    
    def __init__(self, delimiter_style="bracket"):
        """
        Initialize converter with delimiter style
        
        Args:
            delimiter_style: "bracket" -> [ENTITY|TYPE] or "xml" -> <TYPE>ENTITY</TYPE>
        """
        self.delimiter_style = delimiter_style
    
    def parse_bio_tokens(self, tokens: List[str], tags: List[str]) -> str:
        """
        Convert BIO-tagged tokens to formatted string
        
        Args:
            tokens: List of word tokens
            tags: List of BIO tags (B-Person, I-Person, O, etc.)
            
        Returns:
            Formatted string with entity markers
        """
        result = []
        current_entity = []
        current_type = None
        
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                
                if current_entity:
                    result.append(self._format_entity(current_entity, current_type))
                    current_entity = []
                
               
                current_type = tag[2:]  
                current_entity = [token]
                
            elif tag.startswith('I-'):
                
                if current_entity:
                    current_entity.append(token)
                else:
                    
                    current_type = tag[2:]
                    current_entity = [token]
                    
            else: 
                
                if current_entity:
                    result.append(self._format_entity(current_entity, current_type))
                    current_entity = []
                    current_type = None
                
              
                result.append(token)
        
      
        if current_entity:
            result.append(self._format_entity(current_entity, current_type))
        
        return ' '.join(result)
    
    def _format_entity(self, entity_tokens: List[str], entity_type: str) -> str:
        """Format entity with delimiters based on style"""
        entity_text = ' '.join(entity_tokens)
        
        if self.delimiter_style == "bracket":
            return f"[{entity_text}|{entity_type}]"
        elif self.delimiter_style == "xml":
            return f"<{entity_type}>{entity_text}</{entity_type}>"
        elif self.delimiter_style == "asterisk":
            return f"**{entity_text}**({entity_type})"
        else:
            return f"[{entity_text}|{entity_type}]"
    
    def convert_from_conll(self, input_file: Path, output_file: Path):
        """
        Convert CoNLL format file to LLM format
        
        CoNLL format (space separated):
        token1 TAG1
        token2 TAG2
        
        (empty line = sentence boundary)
        """
        sentences = []
        current_tokens = []
        current_tags = []
        
        print(f"Reading from: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:  # Empty line = sentence boundary
                    if current_tokens:
                        formatted = self.parse_bio_tokens(current_tokens, current_tags)
                        sentences.append({
                            'original_tokens': current_tokens.copy(),
                            'bio_tags': current_tags.copy(),
                            'formatted_text': formatted
                        })
                        current_tokens = []
                        current_tags = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[1]
                        current_tokens.append(token)
                        current_tags.append(tag)
                    elif len(parts) == 1:
                        # Handle lines with only token (treat as O)
                        current_tokens.append(parts[0])
                        current_tags.append('O')
        
        # Don't forget last sentence if file doesn't end with empty line
        if current_tokens:
            formatted = self.parse_bio_tokens(current_tokens, current_tags)
            sentences.append({
                'original_tokens': current_tokens,
                'bio_tags': current_tags,
                'formatted_text': formatted
            })
        
        # Save to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Converted {len(sentences)} sentences")
        print(f"✓ Saved to: {output_file}")
        
        # Show statistics
        self._print_statistics(sentences)
        
        return sentences
    
    def _print_statistics(self, sentences: List[Dict]):
        """Print conversion statistics"""
        total_tokens = sum(len(s['original_tokens']) for s in sentences)
        entity_counts = {}
        
        for sentence in sentences:
            for tag in sentence['bio_tags']:
                if tag.startswith('B-'):
                    entity_type = tag[2:]
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        print(f"\n{'='*50}")
        print(f"Statistics:")
        print(f"{'='*50}")
        print(f"Total sentences: {len(sentences)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per sentence: {total_tokens/len(sentences):.1f}")
        print(f"\nEntity counts:")
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count}")
        print(f"{'='*50}\n")
    
    def convert_from_json(self, input_file: Path, output_file: Path):
        """
        Convert JSON format to LLM format
        
        Expected JSON structure:
        [
            {
                "tokens": ["token1", "token2", ...],
                "tags": ["B-Person", "I-Person", "O", ...]
            },
            ...
        ]
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentences = []
        for item in data:
            tokens = item.get('tokens', [])
            tags = item.get('tags', [])
            
            formatted = self.parse_bio_tokens(tokens, tags)
            sentences.append({
                'original_tokens': tokens,
                'bio_tags': tags,
                'formatted_text': formatted
            })
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Converted {len(sentences)} sentences")
        print(f"✓ Saved to: {output_file}")
        
        self._print_statistics(sentences)
        
        return sentences


def main():
    """Convert EverestNER-train-bio file"""
    
    print("\n" + "="*60)
    print("BIO to LLM Format Converter for Nepali NER")
    print("="*60 + "\n")
    
    # Define paths
    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    # Input file
    input_file = data_dir / "EverestNER-train-bio.txt"
    
    if not input_file.exists():
        print(f"❌ Error: File not found - {input_file}")
        print(f"Please make sure 'EverestNER-train-bio' is in the {data_dir} folder")
        return
    
    # Try different delimiter styles
    delimiter_styles = ["bracket", "xml", "asterisk"]
    
    for style in delimiter_styles:
        print(f"\n--- Converting with '{style}' delimiter style ---")
        
        # Initialize converter
        converter = BIOConverter(delimiter_style=style)
        
        # Output file
        output_file = output_dir / f"formatted_train_{style}.json"
        
        
        sentences = converter.convert_from_conll(input_file, output_file)
        
        # Show examples
        print(f"\nExample outputs (first 3 sentences):")
        for i, sent in enumerate(sentences[:3], 1):
            print(f"\n{i}. Original: {' '.join(sent['original_tokens'][:15])}...")
            print(f"   Formatted: {sent['formatted_text'][:100]}...")
    
    print(f"\n{'='*60}")
    print("✓ Conversion complete!")
    print(f"Check the '{output_dir}' folder for output files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
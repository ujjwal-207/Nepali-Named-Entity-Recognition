import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.converter import BIOConverter

def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    converter = BIOConverter(delimiter_style="bracket")
    
    files_to_convert = [
        ("nepali_ner.conll", "formatted_data.json"),
        ("train.json", "formatted_train.json"),
        ("test.json", "formatted_test.json"),
    ]
    
    for input_file, output_file in files_to_convert:
        input_path = raw_dir / input_file
        output_path = processed_dir / output_file
        
        if input_path.exists():
            print(f"\nConverting {input_file}...")
            if input_file.endswith('.conll'):
                converter.convert_from_conll(input_path, output_path)
            elif input_file.endswith('.json'):
                converter.convert_from_json(input_path, output_path)
        else:
            print(f"Skipping {input_file} (not found)")

if __name__ == "__main__":
    main()
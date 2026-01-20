
from pathlib import Path
import random
from typing import List, Tuple


def load_sentences(file_path: Path) -> List[List[str]]:
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def save_sentences(sentences: List[List[str]], file_path: Path):
    """Save sentences to BIO file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n') 


def split_data(
    input_file: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    
    sentences = load_sentences(input_file)
    total = len(sentences)
    
    random.seed(seed)
    random.shuffle(sentences)
    
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    train_data = sentences[:train_size]
    val_data = sentences[train_size:train_size + val_size]
    test_data = sentences[train_size + val_size:]
    
    print(f"  Train: {len(train_data):,} sentences ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} sentences ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} sentences ({len(test_data)/total*100:.1f}%)")
    print()
    
    save_sentences(train_data, output_dir / "EverestNER-train-bio.txt")
    save_sentences(val_data, output_dir / "EverestNER-val-bio.txt")
    save_sentences(test_data, output_dir / "EverestNER-test-bio.txt")
    
    
    
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        entity_count = 0
        for sentence in split_data:
            for line in sentence:
                parts = line.split()
                if len(parts) >= 2 and not parts[1].startswith('O'):
                    entity_count += 1
        
        print(f"{split_name}:")
        print(f"  Sentences: {len(split_data):,}")
        print(f"  Entities:  {entity_count:,}")
        print()
    
    


def main():
    input_file = Path("data/raw/EverestNER-train-bio.txt")
    output_dir = Path("data/raw")
    
    if not input_file.exists():
        print(f"\n Error: {input_file} not found!")
        print("Please check the file path.\n")
        return
    
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.\n")
        return
    
    backup_file = Path("data/raw/EverestNER-train-bio.txt.original")
    if not backup_file.exists():
        import shutil
        shutil.copy(input_file, backup_file)
        print(f"\n Backed up original to: {backup_file}")
    
    split_data(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )


if __name__ == "__main__":
    main()

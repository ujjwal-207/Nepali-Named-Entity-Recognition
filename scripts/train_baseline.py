"""
Script to train baseline NER models - CPU OPTIMIZED
For systems with 8GB RAM and NO GPU

Usage: 
  python scripts/train_baseline.py
  python scripts/train_baseline.py --epochs 3  # Faster test
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.baseline_trainer import NERTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train baseline NER model (CPU Optimized)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-multilingual-cased',
        choices=[
            'bert-base-multilingual-cased', 
            'distilbert-base-multilingual-cased', 
            'xlm-roberta-base',  
        ],
        help='Pre-trained model to fine-tune'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing BIO format training data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/baseline',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size (4 for 8GB RAM)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (5 recommended for CPU)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=64,
        help='Maximum sequence length (64 for low RAM)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with 2 epochs (for testing setup)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    

    if args.quick_test:
        args.epochs = 2
        print("\n⚡ QUICK TEST MODE: Running only 2 epochs")
    
    print("\n" + "="*70)
    print("NEPALI NER BASELINE TRAINING - CPU VERSION")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Batch size: {args.batch_size} (optimized for 8GB RAM)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length} (reduced for memory)")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"  - Uses ORIGINAL BIO files (NOT JSON)")
    print(f"  - Model will download to PC (~500MB)")
    print(f"  - Estimated time: {args.epochs * 1}+ hours on CPU")
    print("="*70)
    

    train_file = Path(args.data_dir) / "EverestNER-train-bio"
    if not train_file.exists():
        print(f"\n❌ ERROR: Training file not found!")
        print(f"Expected location: {train_file}")
        print(f"\nPlease ensure your BIO format file is at this location.")
        print(f"Format should be:")
        print(f"  token1 TAG1")
        print(f"  token2 TAG2")
        print(f"  (empty line)")
        return
    
    print(f"\n✓ Found training file: {train_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = NERTrainer(
        model_name=args.model,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        seed=args.seed
    )
    
    # Training pipeline
    print("\n🚀 Starting training...")
    trainer.load_data()
    trained_model = trainer.train()
    trainer.evaluate(trained_model)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📦 Model saved to:")
    print(f"  {output_dir / 'final_model'}")
    print(f"\n📊 Check results:")
    print(f"  {output_dir / 'test_results.json'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import random
import math
from collections import defaultdict
import wandb
import os

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)




from agents import RankingBasedHRLTrainer


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Ranking-based HRL Trainer for Bug Localization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--reward_type", type=str, choices=["sparse", "intermediate"], default="sparse",
                        help="Type of reward to use: sparse or intermediate")
    parser.add_argument("--wrong_item_penalty", type=float, default=0.1,
                        help="Negative reward for wrong items per ranked position in intermediate mode (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizers")
    parser.add_argument("--training_strategy", type=str, choices=["vt", "pt1", "pt2", "t1b1", "twf", "t1b1rl", "flat"], default="vt",
                        help="Training strategy: vt, pt1, pt2, t1b1, twf, t1b1rl, flat (hierarchy ablation)")
    parser.add_argument("--encoder_type", type=str, choices=["lstm", "mlp"], default="lstm",
                        help="Encoder type for file/line agents: lstm (BiLSTM, default) or mlp (MLP ablation)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="Number of pretraining epochs for pt1 strategy")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--data_dir", type=str, default="../../data/swebench/hierarchical_dataset",
                        help="Path to directory containing train_filtered.json and val_filtered.json")
    parser.add_argument("--checkpoint_dir", type=str, default="../../outputs/checkpoints/rl/swebench",
                        help="Directory for saving model checkpoints")
    parser.add_argument("--embedder_ckpt_path", type=str,
                        default="../../outputs/checkpoints/contrastive/swebench/hardneg/best_codebert_triplet.pt",
                        help="Path to CodeBERT checkpoint used for on-the-fly embeddings (must match the model used to build stack_trace_embedding in the dataset)")
    parser.add_argument("--tokenizer_path", type=str,
                        default="../../models/codebert/codebert_tokenizer",
                        help="Path to CodeBERT tokenizer directory")
    parser.add_argument("--use_path", action="store_true",
                        help="Prepend file path tokens to code chunks (must match contrastive training)")
    return parser.parse_args()



if __name__ == "__main__":
    import json
    from pathlib import Path
    from dataembedder import CodeBERTEmbedder
    
    # Parse arguments
    args = parse_args()
    
    # Fix seed for reproducibility
    fix_seed(args.seed)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training strategy: {args.training_strategy}")
    
    # Load dataset — use filtered versions which exclude entries with
    # correct_file_idx == -1 (bug not in FAISS index). Unfiltered data causes
    # pretrain strategies to silently train toward wrong targets via negative
    # index wrapping (Python: list[-1] = last element; PyTorch: cross_entropy
    # target=-1 wraps to last class).
    dataset_path = Path(args.data_dir) / "train_filtered.json"
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)

    # Use val set for in-training evaluation and best-checkpoint selection.
    # test set is reserved exclusively for final evaluation via evaluate.py.
    val_dataset_path = Path(args.data_dir) / "val_filtered.json"
    with open(val_dataset_path, 'r') as f:
        val_data = json.load(f)

    print(f"Training samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Initialize embedder — must use same checkpoint as was used to build stack_trace_embedding
    embedder = CodeBERTEmbedder(tokenizer_path=args.tokenizer_path, model_ckpt_path=args.embedder_ckpt_path, device=device, use_path=args.use_path)

    # Initialize and train with specified strategy
    trainer = RankingBasedHRLTrainer(
        train_data=train_data,
        test_data=val_data,
        embedder=embedder,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        file_top_k=5,
        func_top_k=15,
        line_top_k=15,
        exploration_mode="epsilon_greedy",
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        reward_type=args.reward_type,
        wrong_item_penalty=args.wrong_item_penalty,
        use_wandb=True,
        project_name="ranking-hrl-bug-localization",
        seed=args.seed,
        training_strategy=args.training_strategy,
        checkpoint_dir=args.checkpoint_dir,
        encoder_type=args.encoder_type
    )
    
    # Train the model with specified strategy
    trainer.train(epochs=args.epochs, eval_interval=args.eval_interval, pretrain_epochs=args.pretrain_epochs)
    
    # Final evaluation
    final_metrics = trainer.evaluate(quick_eval=False)
    print(f"Final Evaluation Metrics: {final_metrics}")
    
    # Log final metrics
    if trainer.use_wandb:
        wandb.log({
            **{f"final/{k}": v for k, v in final_metrics.items() if isinstance(v, (int, float))}
        })
        wandb.finish()
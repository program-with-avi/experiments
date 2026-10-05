"""Training Pipeline for ESP32 Generative Conversational Language Model.

Features:
- Residual LayerNorm GRU (LNResGRUModel) architecture (~433K params).
- Masked Cross-Entropy Loss: Computes loss strictly on the target bot response tokens,
  ignoring user prompts, turn markers, and padding tokens.
- AdamW Optimizer with gradient clipping and Cosine Annealing learning rate schedule.
- Validation loss and perplexity tracking per epoch.
- Best model checkpoint saving with full architectural metadata.
- Automated multi-topic qualitative evaluation across diverse conversational prompts.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure chatbot_lib is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from chatbot_lib.tokenizer import MicroTokenizer
from chatbot_lib.dataset import generate_conversational_corpus, create_dataloaders
from chatbot_lib.model import LNResGRUModel
from chatbot_lib.generator import TextGenerator


def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int
) -> Tuple[torch.Tensor, int]:
    """Computes cross-entropy loss exclusively on supervised tokens (mask == 1.0).
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        targets: Tensor of shape (batch_size, seq_len)
        mask: Float tensor of shape (batch_size, seq_len) with 1.0 on bot tokens
        vocab_size: Size of vocabulary
        
    Returns:
        loss: Scalar tensor representing average loss per supervised token
        token_count: Number of supervised tokens in this batch
    """
    flat_logits = logits.view(-1, vocab_size)
    flat_targets = targets.view(-1)
    flat_mask = mask.view(-1)

    # Compute unreduced cross-entropy for each token
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

    # Mask out prompt tokens and padding
    masked_loss = per_token_loss * flat_mask
    num_tokens = flat_mask.sum().item()

    if num_tokens > 0:
        loss = masked_loss.sum() / (flat_mask.sum() + 1e-8)
    else:
        loss = masked_loss.sum()

    return loss, int(num_tokens)


def evaluate(
    model: LNResGRUModel,
    val_loader,
    device: torch.device,
    vocab_size: int
) -> Tuple[float, float]:
    """Evaluates model on validation dataset, returning loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y, mask in val_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(x)
            loss, num_tokens = compute_masked_loss(logits, y, mask, vocab_size)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


def train(args):
    """Main training routine."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"=== Starting Generative AI Model Training ===")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # 1. Load Tokenizer
    print(f"\n[1/5] Loading MicroTokenizer from '{args.tokenizer_path}'...")
    tokenizer = MicroTokenizer.load(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size = {tokenizer.vocab_size} (special={tokenizer.NUM_SPECIAL})")

    # 2. Prepare Dataset
    print(f"\n[2/5] Synthesizing conversational corpus (target_count={args.dataset_size})...")
    corpus = generate_conversational_corpus(
        seed=args.seed,
        augment_variations=True,
        target_count=args.dataset_size
    )
    print(f"Corpus generated with {len(corpus)} dialogue pairs across 35 topics.")

    train_loader, val_loader, ds_stats = create_dataloaders(
        dialogue_pairs=corpus,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_length=args.max_length,
        seed=args.seed
    )
    print(f"Dataset split: Train = {ds_stats['train_pairs']} pairs ({ds_stats['num_train_batches']} batches), "
          f"Val = {ds_stats['val_pairs']} pairs ({ds_stats['num_val_batches']} batches)")

    # 3. Initialize Generative Model
    print(f"\n[3/5] Instantiating LNResGRUModel...")
    model = LNResGRUModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=True
    )
    model.to(device)

    diag = model.get_diagnostics()
    print(f"Model Architecture:")
    print(f"  - Vocab Size: {diag['vocab_size']}")
    print(f"  - Hidden Size: {diag['hidden_size']}")
    print(f"  - Layers: {diag['num_layers']}")
    print(f"  - Tied Embeddings: {diag['tie_weights']}")
    print(f"  - Trainable Parameters: {diag['trainable_params']:,} (~{diag['trainable_params']/1000:.1f}K)")
    print(f"  - FP32 Size: {diag['fp32_size_kb']:.2f} KB | INT8 Size: {diag['int8_size_kb']:.2f} KB")
    print(f"  - ESP32 Activation SRAM: {diag['sram_hidden_state_kb']:.3f} KB")
    print(f"  - Feasibility Check: Flash={diag['esp32_flash_feasible']}, SRAM={diag['esp32_sram_feasible']}")

    # 4. Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )

    best_val_loss = float("inf")
    history: List[Dict] = []
    t_start = time.time()

    print(f"\n[4/5] Training for {args.epochs} epochs with masked cross-entropy...")
    print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Train PPL':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'LR':<9} | {'Time':<6}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_tokens = 0

        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss, num_tokens = compute_masked_loss(
                logits, y, mask, tokenizer.vocab_size
            )
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * num_tokens
            total_train_tokens += num_tokens

        train_loss = total_train_loss / max(total_train_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20.0))

        # Validation step
        val_loss, val_ppl = evaluate(model, val_loader, device, tokenizer.vocab_size)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"{epoch:<7} | {train_loss:<12.4f} | {train_ppl:<10.2f} | {val_loss:<10.4f} | {val_ppl:<10.2f} | {current_lr:<9.6f} | {epoch_time:.1f}s")

        # Save checkpoint if best validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            meta = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "dataset_size": args.dataset_size,
                "timestamp": time.time()
            }
            model.save_checkpoint(args.save_path, extra_meta=meta)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "lr": round(current_lr, 6),
            "time_sec": round(epoch_time, 2)
        })

    total_training_time = time.time() - t_start
    print(f"\nTraining completed in {total_training_time:.1f}s.")
    print(f"Best Validation Loss: {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 20.0)):.2f})")
    print(f"Best checkpoint saved to '{args.save_path}'.")

    # 5. Qualitative Multi-Topic Evaluation
    print(f"\n[5/5] Generating qualitative sample responses across conversational topics...")
    # Load best checkpoint for generation
    best_model, meta = LNResGRUModel.load_checkpoint(args.save_path, device=device)
    generator = TextGenerator(best_model, tokenizer, device=device)

    test_prompts = [
        "who are you?",
        "what is your purpose?",
        "what is the sun?",
        "why is the sky blue?",
        "what is an esp32?",
        "what is an ESP32?",
        "tell me about Python.",
        "what is artificial intelligence?",
        "tell me about trees.",
        "why do people love music?",
        "hello friend!",
        "what is friendship?",
        "do you need to sleep?",
        "what is a tokenizer?"
    ]

    sample_generations = []
    print("\n" + "=" * 70)
    print("QUALITATIVE TEXT GENERATION SAMPLES")
    print("=" * 70)

    for prompt in test_prompts:
        resp_greedy = generator.generate(
            prompt, max_new_tokens=48, temperature=0.0
        )
        resp_sampled = generator.generate(
            prompt, max_new_tokens=48, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.15
        )

        sample_generations.append({
            "prompt": prompt,
            "greedy_response": resp_greedy,
            "sampled_response": resp_sampled
        })

        print(f"\nUser: {prompt}")
        print(f"Bot (Greedy):  {resp_greedy}")
        print(f"Bot (Sampled): {resp_sampled}")

    print("\n" + "=" * 70)

    # Save summary report
    summary = {
        "model_type": "LNResGRUModel",
        "trainable_parameters": diag["trainable_params"],
        "fp32_size_kb": diag["fp32_size_kb"],
        "int8_size_kb": diag["int8_size_kb"],
        "sram_hidden_state_kb": diag["sram_hidden_state_kb"],
        "esp32_feasible": diag["esp32_flash_feasible"] and diag["esp32_sram_feasible"],
        "final_train_loss": history[-1]["train_loss"],
        "final_train_ppl": history[-1]["train_ppl"],
        "best_val_loss": round(best_val_loss, 4),
        "best_val_ppl": round(math.exp(min(best_val_loss, 20.0)), 2),
        "epochs": args.epochs,
        "total_training_time_sec": round(total_training_time, 2),
        "dataset_size": args.dataset_size,
        "history": history,
        "sample_generations": sample_generations
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.summary_path)), exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Milestone 3 summary saved to '{args.summary_path}'.")

    return summary


def parse_args():
    default_tok = os.path.join(current_dir, "data", "tokenizer_768.json")
    default_save = os.path.join(current_dir, "data", "generative_model.pt")
    default_summary = os.path.join(current_dir, "data", "milestone3_summary.json")

    parser = argparse.ArgumentParser(description="Train ESP32 Generative Model")
    parser.add_argument("--tokenizer-path", type=str, default=default_tok)
    parser.add_argument("--save-path", type=str, default=default_save)
    parser.add_argument("--summary-path", type=str, default=default_summary)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

import argparse
import ast
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, RobertaModel, get_scheduler
from torch.optim import AdamW
import json
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import random
import numpy as np
from torch.amp import autocast, GradScaler
import torch.nn as nn


class _StripDocstrings(ast.NodeTransformer):
    """AST transformer that removes docstrings from modules, classes, and functions."""
    def _strip(self, node):
        self.generic_visit(node)
        if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
        return node
    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def filter_python_for_embedding(source):
    """Remove # comments and docstrings from Python source via AST round-trip.
    Falls back to raw source on SyntaxError (e.g. Python 2 syntax, encoding issues)."""
    try:
        tree = ast.parse(source)
        tree = _StripDocstrings().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return source


def filter_java_for_embedding(text):
    """Strip block comments, line comments, imports, package decls, and blank lines from Java source."""
    if not text:
        return text
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    lines = text.split('\n')
    filtered = []
    for line in lines:
        s = line.strip()
        if not s:                                       continue
        if s.startswith('//'):                          continue
        if s.startswith('import '):                     continue
        if s.startswith('package '):                    continue
        filtered.append(line)
    return '\n'.join(filtered)


def filter_for_embedding(text, filepath=""):
    """Route to Python or Java filter by file extension; returns text unchanged for other types."""
    if filepath.endswith(".py"):
        return filter_python_for_embedding(text)
    if filepath.endswith(".java"):
        return filter_java_for_embedding(text)
    return text


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class TripletDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        positive_path = item.get('positive_path', '')
        negative_path = item.get('negative_path', '')
        return item['anchor'], item['positive'], item['negative'], item['issue_id'], positive_path, negative_path

def collate_fn(batch):
    anchors, positives, negatives, issue_ids, pos_paths, neg_paths = zip(*batch)
    return list(anchors), list(positives), list(negatives), list(issue_ids), list(pos_paths), list(neg_paths)


class CodeBERTTriplet(nn.Module):
    def __init__(self, model_path=None, device="cuda", use_path=False, max_chunks=50):
        super().__init__()
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.device = torch.device(device)
        self.use_path = use_path
        self.max_chunks = max_chunks

        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if model_path:
            print(f"Loading weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            cleaned = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            self.model.load_state_dict(cleaned)

    def sliding_window_encode(self, text, path="", max_tokens=512, device=None):
        device = device or self.device
        path_ids = self.tokenizer.encode(f"# {path}", add_special_tokens=False) if path else []
        content_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not content_ids:
            return F.normalize(torch.zeros(1, self.model.config.hidden_size, device=device), dim=1)
        # Each chunk: [CLS] path_ids content_chunk [SEP], content budget shrinks by len(path_ids).
        # max(1, ...) ensures we never get a zero-step range for unusually long paths.
        content_budget = max(1, max_tokens - 2 - len(path_ids))
        chunks = [content_ids[i:i + content_budget] for i in range(0, len(content_ids), content_budget)]
        chunks = chunks[:self.max_chunks]
        all_ids, all_masks = [], []
        for chunk in chunks:
            ids = [self.tokenizer.cls_token_id] + path_ids + chunk + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)
            pad_len = max_tokens - len(ids)
            ids  += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len
            all_ids.append(ids)
            all_masks.append(mask)
        inputs = {
            "input_ids":      torch.tensor(all_ids,   device=device),
            "attention_mask": torch.tensor(all_masks, device=device),
        }
        cls = self.model(**inputs).last_hidden_state[:, 0, :]  # (num_chunks, 768)
        return F.normalize(cls.mean(dim=0).unsqueeze(0), dim=1)

    def encode_batch(self, texts, device=None, paths=None):
        device = device or self.device
        if paths is None:
            paths = [""] * len(texts)
        return torch.cat([self.sliding_window_encode(t, path=p, device=device) for t, p in zip(texts, paths)], dim=0)

    def forward(self, anchor, positive, negative, pos_paths=None, neg_paths=None):
        device = self.device
        _pp = pos_paths if pos_paths else [""] * len(positive)
        _np = neg_paths if neg_paths else [""] * len(negative)
        filtered_pos = [filter_for_embedding(t, p) for t, p in zip(positive, _pp)]
        filtered_neg = [filter_for_embedding(t, p) for t, p in zip(negative, _np)]
        return (
            self.encode_batch(anchor, device),
            self.encode_batch(filtered_pos, device, paths=_pp if self.use_path else None),
            self.encode_batch(filtered_neg, device, paths=_np if self.use_path else None),
        )


def build_issue_mask(issue_ids, device, pos_paths=None):
    # Map issue IDs to consecutive ints — supports both int and string IDs (e.g. SWE-bench)
    unique = list(dict.fromkeys(issue_ids))
    id_to_int = {iid: i for i, iid in enumerate(unique)}
    int_ids = [id_to_int[iid] for iid in issue_ids]
    ids_tensor = torch.tensor(int_ids, dtype=torch.long, device=device)
    same_issue = (ids_tensor.unsqueeze(0) == ids_tensor.unsqueeze(1))

    # Also mask same-positive-file pairs from different issues: if two issues
    # both edit the same file, using one as an in-batch negative for the other
    # is a false negative (the positive embeddings are nearly identical).
    if pos_paths is not None:
        unique_paths = list(dict.fromkeys(pos_paths))
        path_to_int = {p: i for i, p in enumerate(unique_paths)}
        int_paths = [path_to_int[p] for p in pos_paths]
        paths_tensor = torch.tensor(int_paths, dtype=torch.long, device=device)
        same_path = (paths_tensor.unsqueeze(0) == paths_tensor.unsqueeze(1))
        exclude = same_issue | same_path
    else:
        exclude = same_issue

    not_self = ~torch.eye(len(issue_ids), dtype=torch.bool, device=device)
    return (~(exclude & not_self)).float()


def hybrid_triplet_loss(a, p, n, issue_ids, temperature=0.1, symmetric=False, pos_paths=None):
    mask = build_issue_mask(issue_ids, a.device, pos_paths=pos_paths)
    labels = torch.arange(a.size(0), device=a.device)

    logits_ap = torch.matmul(a, p.T) / temperature
    masked_ap = logits_ap * mask + (1 - mask) * -1e9
    inbatch_loss = F.cross_entropy(masked_ap, labels)

    if symmetric:
        logits_pa = torch.matmul(p, a.T) / temperature
        masked_pa = logits_pa * mask + (1 - mask) * -1e9
        inbatch_loss = (inbatch_loss + F.cross_entropy(masked_pa, labels)) / 2

    pos_sim = (a * p).sum(dim=1)
    neg_sim = (a * n).sum(dim=1)
    margin = 0.2
    explicit_loss = F.relu(margin + neg_sim - pos_sim).mean()

    return inbatch_loss + explicit_loss, inbatch_loss.item(), explicit_loss.item(), pos_sim.mean().item(), neg_sim.mean().item()


def evaluate(model, dataloader, device, symmetric=False):
    model.eval()
    sim_scores, neg_scores = [], []
    val_loss_total = 0
    correct_top1 = correct_top3 = correct_top5 = total = 0

    with torch.no_grad():
        for anchor, pos, neg, issue_ids, pos_paths, neg_paths in dataloader:
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                a, p, n = model(anchor, pos, neg, pos_paths=pos_paths, neg_paths=neg_paths)

            all_candidates = torch.cat([p, n], dim=0)
            sims = torch.matmul(a, all_candidates.T)
            topk = sims.topk(5, dim=1).indices
            correct = torch.arange(a.size(0), device=device).unsqueeze(1)
            correct_top1 += (topk[:, :1] == correct).sum().item()
            correct_top3 += (topk[:, :3] == correct).any(dim=1).sum().item()
            correct_top5 += (topk[:, :5] == correct).any(dim=1).sum().item()

            val_loss, _, _, pos_sim, neg_sim = hybrid_triplet_loss(a, p, n, issue_ids, symmetric=symmetric, pos_paths=pos_paths)
            val_loss_total += val_loss.item()
            sim_scores.append(pos_sim)
            neg_scores.append(neg_sim)
            total += a.size(0)

    avg_sim = sum(sim_scores) / len(sim_scores)
    avg_neg_sim = sum(neg_scores) / len(neg_scores)
    acc_top1 = correct_top1 / total
    acc_top3 = correct_top3 / total
    acc_top5 = correct_top5 / total
    avg_val_loss = val_loss_total / len(dataloader)

    wandb.log({
        "eval_avg_pos_similarity": avg_sim,
        "eval_avg_neg_similarity": avg_neg_sim,
        "eval_top1_accuracy": acc_top1,
        "eval_top3_accuracy": acc_top3,
        "eval_top5_accuracy": acc_top5,
        "eval_loss": avg_val_loss
    })
    print(f"Eval - PosSim: {avg_sim:.4f}, NegSim: {avg_neg_sim:.4f}, "
          f"Top-1: {acc_top1:.4f}, Top-3: {acc_top3:.4f}, Top-5: {acc_top5:.4f}, Loss: {avg_val_loss:.4f}")
    return avg_val_loss, acc_top1, acc_top3, acc_top5


def train(model, dataloader, optimizer, scheduler, scaler, device, epoch, iters_to_accumulate, symmetric=False):
    model.train()
    total_loss = 0
    total_pos_sim = 0
    total_neg_sim = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        anchor, pos, neg, issue_ids, pos_paths, neg_paths = batch
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            a, p, n = model(anchor, pos, neg, pos_paths=pos_paths, neg_paths=neg_paths)
            loss, inbatch, explicit, pos_sim, neg_sim = hybrid_triplet_loss(a, p, n, issue_ids, symmetric=symmetric, pos_paths=pos_paths)
            scaled_loss = loss / iters_to_accumulate

        scaler.scale(scaled_loss).backward()

        if (step + 1) % iters_to_accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()
        total_pos_sim += pos_sim
        total_neg_sim += neg_sim

        wandb.log({
            "batch_loss": loss.item(),
            "inbatch_loss": inbatch,
            "explicit_loss": explicit,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "epoch": epoch + 1,
            "step": step
        })

    # Flush any leftover accumulated gradients
    if (step + 1) % iters_to_accumulate != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    n_steps = len(dataloader)
    wandb.log({
        "epoch_avg_loss":    total_loss    / n_steps,
        "epoch_avg_pos_sim": total_pos_sim / n_steps,
        "epoch_avg_neg_sim": total_neg_sim / n_steps,
        "epoch": epoch + 1,
    })
    return total_loss / n_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--triplet_path", type=str, default="../../data/triplet_dataset.json", help="Path to triplet data")
    parser.add_argument("--output_dir", type=str, default="../../outputs/checkpoints/contrastive", help="Directory for saving model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (use lower e.g. 5e-6 for warm-start)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--warmup_steps", type=float, default=0.1, help="LR warmup: if < 1, fraction of total steps (default 0.1 = 10%%); if >= 1, absolute count; use 0 for warm-start")
    parser.add_argument("--use_path", action="store_true", default=False, help="Enable path-aware file embeddings")
    parser.add_argument("--max_chunks", type=int, default=50, help="Max sliding window chunks per text (reduce to save GPU memory)")
    parser.add_argument("--symmetric_loss", action="store_true", default=False, help="Add symmetric p->a InfoNCE direction to in-batch loss")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Freeze bottom N transformer encoder layers (0 = no freezing). Cite: Telly et al. ISSTA 2023.")
    parser.add_argument("--selection_metric", type=str, default="loss", choices=["loss", "top1", "top3", "top5"],
                        help="Metric for best model selection and early stopping: 'loss' (lower=better, default, good for random negatives) or 'top1/3/5' (higher=better, recommended for hard negatives).")
    parser.add_argument("--wandb_project", type=str, default="c2c-codebert",
                        help="Weights & Biases project name.")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name="fine-tune-codebert")
    dataset = TripletDataset(args.triplet_path)

    train_indices = [i for i, item in enumerate(dataset.data) if item.get("split") == "train"]
    val_indices = [i for i, item in enumerate(dataset.data) if item.get("split") == "val"]
    excluded = len(dataset) - len(train_indices) - len(val_indices)
    print(f"Issue-level split: {len(train_indices)} train triplets, "
          f"{len(val_indices)} val triplets (excluded {excluded} test triplets)")
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeBERTTriplet(model_path=args.model_path, device=str(device), use_path=args.use_path, max_chunks=args.max_chunks).to(device)

    if args.freeze_layers > 0:
        n_layers = len(model.model.encoder.layer)
        n_freeze = min(args.freeze_layers, n_layers)
        for layer in model.model.encoder.layer[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Froze bottom {n_freeze}/{n_layers} transformer layers. "
              f"Trainable: {trainable:,} / {total:,} params")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    iters_to_accumulate = args.grad_accum
    num_updates_per_epoch = len(train_loader) // iters_to_accumulate + 1
    num_training_steps = num_updates_per_epoch * args.epochs
    if args.warmup_steps < 1:
        num_warmup_steps = int(num_training_steps * args.warmup_steps)
    else:
        num_warmup_steps = int(args.warmup_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=num_warmup_steps,
                              num_training_steps=num_training_steps)
    print(f"Scheduler: {num_training_steps} total steps "
          f"({num_updates_per_epoch}/epoch), warmup={num_warmup_steps}")

    use_loss = args.selection_metric == "loss"
    best_score = float('inf') if use_loss else -1.0
    patience = args.patience
    counter = 0
    scaler = GradScaler()

    for epoch in range(args.epochs):
        print(f"\n==== Epoch {epoch + 1} ====")
        train(model, train_loader, optimizer, scheduler, scaler, device, epoch, iters_to_accumulate, symmetric=args.symmetric_loss)
        torch.cuda.empty_cache()
        current_loss, current_top1, current_top3, current_top5 = evaluate(model, val_loader, device, symmetric=args.symmetric_loss)
        metric_map = {"loss": current_loss, "top1": current_top1, "top3": current_top3, "top5": current_top5}
        current_score = metric_map[args.selection_metric]
        improved = current_score < best_score if use_loss else current_score > best_score
        print(f"Validation Loss: {current_loss:.4f} | Val Top-3 Acc: {current_top3:.4f} | [{args.selection_metric}={current_score:.4f}]")
        if improved:
            best_score = current_score
            counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_codebert_triplet.pt"))
            print(f"Saved best model ({args.selection_metric}={best_score:.4f}).")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()

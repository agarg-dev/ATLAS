import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
        return item['anchor'], item['positive'], item['negative'], item['issue_id']

def collate_fn(batch, tokenizer=None):
    anchors, positives, negatives, issue_ids = zip(*batch)
    return list(anchors), list(positives), list(negatives), list(issue_ids)


class CodeBERTTriplet(nn.Module):
    def __init__(self, model_path=None, device="cuda"):
        super().__init__()
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.device = torch.device(device)

        if model_path:
            print(f"Loading weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned)

    def sliding_window_encode(self, text, max_tokens=512, stride=510, device=None):
        device = device or self.device
        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not input_ids:
            return F.normalize(torch.zeros(1, self.model.config.hidden_size, device=device), dim=1)
        chunks = [input_ids[i:i + max_tokens - 2] for i in range(0, len(input_ids), stride)]
        chunks = chunks[:12]
        embeddings = []
        for chunk in chunks:
            ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)
            pad_len = max_tokens - len(ids)
            ids += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len
            inputs = {
                "input_ids": torch.tensor([ids], device=device),
                "attention_mask": torch.tensor([mask], device=device)
            }
            cls = self.model(**inputs).last_hidden_state[:, 0, :]
            embeddings.append(cls.squeeze(0))

        stacked = torch.stack(embeddings)
        mean_pooled = stacked.mean(dim=0)
        return F.normalize(mean_pooled.unsqueeze(0), dim=1)

    def encode_batch(self, texts, device=None):
        device = device or self.device
        return torch.cat([self.sliding_window_encode(t, device=device) for t in texts], dim=0)

    def forward(self, anchor, positive, negative):
        device = self.device
        return (
            self.encode_batch(anchor, device),
            self.encode_batch(positive, device),
            self.encode_batch(negative, device)
        )


def build_issue_mask(issue_ids, device):
    ids_tensor = torch.tensor(issue_ids, dtype=torch.long, device=device)
    same_issue = (ids_tensor.unsqueeze(0) == ids_tensor.unsqueeze(1))
    not_self = ~torch.eye(len(issue_ids), dtype=torch.bool, device=device)
    mask = ~(same_issue & not_self)
    return mask.float()


def hybrid_triplet_loss(a, p, n, issue_ids, temperature=0.1):
    logits_inbatch = torch.matmul(a, p.T) / temperature
    mask = build_issue_mask(issue_ids, a.device)
    masked_logits = logits_inbatch * mask + (1 - mask) * -1e9

    labels = torch.arange(a.size(0), device=a.device)
    inbatch_loss = F.cross_entropy(masked_logits, labels)

    pos_sim = (a * p).sum(dim=1)
    neg_sim = (a * n).sum(dim=1)
    margin = 0.2
    explicit_loss = F.relu(margin + neg_sim - pos_sim).mean()

    return inbatch_loss + explicit_loss, inbatch_loss.item(), explicit_loss.item(), pos_sim.mean().item(), neg_sim.mean().item()


def evaluate(model, dataloader, device):
    model.eval()
    sim_scores, neg_scores = [], []
    val_loss_total = 0
    correct_top1 = correct_top3 = correct_top5 = total = 0

    with torch.no_grad():
        for anchor, pos, neg, issue_ids in dataloader:
            a = model.encode_batch(anchor, device)
            p = model.encode_batch(pos, device)
            n = model.encode_batch(neg, device)

            all_candidates = torch.cat([p, n], dim=0)
            sims = torch.matmul(a, all_candidates.T)
            topk = sims.topk(5, dim=1).indices
            correct = torch.arange(a.size(0), device=device).unsqueeze(1)
            correct_top1 += (topk[:, :1] == correct).sum().item()
            correct_top3 += (topk[:, :3] == correct).any(dim=1).sum().item()
            correct_top5 += (topk[:, :5] == correct).any(dim=1).sum().item()

            val_loss, _, _, pos_sim, neg_sim = hybrid_triplet_loss(a, p, n, issue_ids)
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
    return avg_val_loss


def train(model, dataloader, optimizer, scheduler, device, epoch, iters_to_accumulate):
    model.train()
    total_loss = 0
    total_pos_sim = 0
    total_neg_sim = 0
    optimizer.zero_grad()
    scaler = GradScaler()
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        anchor, pos, neg, issue_ids = batch
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            a, p, n = model(anchor, pos, neg)
            loss, inbatch, explicit, pos_sim, neg_sim = hybrid_triplet_loss(a, p, n, issue_ids)
            loss = loss / iters_to_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()
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

    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch + 1})
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--triplet_path", type=str, default="../data/triplet_dataset.json", help="Path to triplet data")
    args = parser.parse_args()

    wandb.init(project="codebert-triplet", name="fine-tune-codebert")
    dataset = TripletDataset(args.triplet_path)
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    data_collator = lambda batch: collate_fn(batch)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeBERTTriplet(model_path=args.model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*10)

    best_loss = float('inf')
    patience = 3
    counter = 0
    iters_to_accumulate = 8

    for epoch in range(5):
        print(f"\n==== Epoch {epoch + 1} ====")
        train(model, train_loader, optimizer, scheduler, device, epoch, iters_to_accumulate)
        torch.cuda.empty_cache()
        current_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {current_loss:.4f}")
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
            torch.save(model.state_dict(), "best_codebert_triplet.pt")
            print("Saved best model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()

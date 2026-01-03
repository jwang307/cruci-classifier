#!/usr/bin/env python3
"""
Fine-tune ESM-2 35 M on labelled sequences.

Run:
    WANDB_PROJECT=cruci python train.py \
        --train_csv data/train.csv \
        --test_csv  data/test.csv

Requires `wandb login` beforehand.
"""
import argparse, random, wandb, torch, pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import esm
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

# ---------------- Dataset ---------------- #
class SeqDataset(Dataset):
    """Dataset wrapper that converts sequences and labels from a CSV file."""

    def __init__(self, csv_path: str, alphabet: esm.data.Alphabet) -> None:
        """Load and store sequence data with associated labels."""
        df = pd.read_csv(csv_path)
        self.labels: torch.Tensor = torch.tensor(df["label"].values, dtype=torch.float32)
        self.seqs: Sequence[str] = df["sequence"].tolist()
        self.batch_converter = alphabet.get_batch_converter()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """Return a single sequence and label pair."""
        return self.seqs[idx], self.labels[idx]

    def collate_fn(self, batch: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of sequences into token tensors and stacked labels."""
        seqs, labels = zip(*batch)
        _, _, tokens = self.batch_converter(list(zip(range(len(seqs)), seqs)))
        return tokens, torch.stack(labels)

# -------------- Model -------------------- #
class ESMClassifier(nn.Module):
    """Binary classifier that fine-tunes representations from ESM-2 35M."""

    def __init__(self) -> None:
        """Initialise the ESM encoder and projection head."""
        super().__init__()
        self.esm, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.hidden = self.esm.embed_dim
        self.classifier = nn.Linear(self.hidden, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of tokenised sequences."""
        with torch.no_grad():
            out = self.esm(tokens, repr_layers=[12], return_contacts=False)
        cls_emb = out["representations"][12][:, 0, :]   # BOS token
        return self.classifier(cls_emb).squeeze(-1)

# ------------- Train / Eval -------------- #
def step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a forward and loss computation step."""
    tokens, labels = (x.to(device) for x in batch)
    logits = model(tokens)
    loss = criterion(logits, labels)
    return loss, logits.detach().cpu(), labels.cpu()

def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.device],
    return_details: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """Evaluate the model, returning metrics and optionally raw predictions."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            _, logits, labels = step(model, batch, nn.BCEWithLogitsLoss(), device)
            all_logits.append(torch.sigmoid(logits))
            all_labels.append(labels)
    probs = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "acc": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "auroc": roc_auc_score(y_true, probs),
    }
    if return_details:
        return metrics, {"probs": probs, "labels": y_true, "preds": preds}
    return metrics

def main(args: argparse.Namespace) -> None:
    """Train the classifier, periodically saving checkpoints and reporting metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMClassifier().to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # data
    train_ds_full = SeqDataset(args.train_csv, model.alphabet)
    train_len = int(0.9 * len(train_ds_full))
    val_len = len(train_ds_full) - train_len
    train_ds, val_ds = random_split(
        train_ds_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )
    test_ds = SeqDataset(args.test_csv, model.alphabet)

    collate = train_ds_full.collate_fn  # same converter
    loaders = {
        "train": DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate),
    }

    wandb.init(project="cruci", entity="jwang003", name="esm2_35m_cruci_classifier", config=vars(args))
    best_f1: float = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss, _, _ = step(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loaders["train"])

        # validation
        metrics = eval_epoch(model, loaders["val"], device)
        wandb.log({"epoch": epoch, "train_loss": epoch_loss, **{f"val_{k}": v for k, v in metrics.items()}})
        print(f"Epoch {epoch} | loss {epoch_loss:.4f} | val_f1 {metrics['f1']:.3f}")

        if metrics["f1"] > best_f1:
            best_f1, best_state = metrics["f1"], model.state_dict()

        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    # -------- Test set -------- #
    if best_state is not None:
        torch.save(best_state, checkpoint_dir / "best.pt")
        model.load_state_dict(best_state)
    test_metrics = eval_epoch(model, loaders["test"], device)
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    print("\nTEST METRICS")
    for k, v in test_metrics.items():
        print(f"{k:10s}: {v:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv",  type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--checkpoint_dir", type=str, default="/large_storage/hielab/jwang/cruci/")
    p.add_argument("--save_every", type=int, default=5)
    args = p.parse_args()
    main(args)

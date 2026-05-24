#!/usr/bin/env python3
"""
Train a lightweight binary classification head on frozen ESM-2 35M features.

Run:
    WANDB_PROJECT=cruci python classifier/train.py \
        --train_csv data/train.csv \
        --test_csv  data/test.csv

Requires `wandb login` beforehand.
"""
import argparse, copy, math, random, wandb, torch, pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import esm
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

try:
    from data_utils import clean_protein_sequence
    from metrics import binary_metrics
except ImportError:  # Allows importing as classifier.train.
    from classifier.data_utils import clean_protein_sequence
    from classifier.metrics import binary_metrics


def resolve_device(device_arg: str) -> torch.device:
    """Resolve auto/cuda/mps/cpu into a torch device."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device_arg != "auto":
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if device.type == "mps" and not mps_available:
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set RNG seeds used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _format_metric(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "nan" if math.isnan(float(value)) else f"{float(value):.4f}"
    return str(value)


# ---------------- Dataset ---------------- #
class SeqDataset(Dataset):
    """Dataset wrapper that converts sequences and labels from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        alphabet: esm.data.Alphabet,
        *,
        clean_sequences: bool = True,
        stop_action: str = "remove",
        invalid_action: str = "replace_x",
        drop_empty: bool = True,
        max_examples: Optional[int] = None,
    ) -> None:
        """Load and store sequence data with associated labels."""
        df = pd.read_csv(csv_path)
        required = {"sequence", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        if max_examples is not None:
            df = df.head(max_examples)

        ids: list[str] = []
        labels: list[float] = []
        seqs: list[str] = []
        changed = dropped = stops_removed = stops_replaced = 0
        invalid_removed = invalid_replaced = 0
        id_values = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]
        for identifier, sequence, label in zip(id_values, df["sequence"].tolist(), df["label"].tolist()):
            if clean_sequences:
                cleaned = clean_protein_sequence(
                    sequence,
                    stop_action=stop_action,
                    invalid_action=invalid_action,
                )
                if cleaned.changed:
                    changed += 1
                stops_removed += cleaned.stops_removed
                stops_replaced += cleaned.stops_replaced
                invalid_removed += cleaned.invalid_removed
                invalid_replaced += cleaned.invalid_replaced
                sequence = cleaned.sequence
            else:
                sequence = str(sequence)
            if not sequence and drop_empty:
                dropped += 1
                continue
            ids.append(identifier)
            labels.append(float(label))
            seqs.append(sequence)

        if not seqs:
            raise ValueError(f"{csv_path} produced no usable sequences")

        self.ids: Sequence[str] = ids
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.float32)
        self.seqs: Sequence[str] = seqs
        self.cleaning_summary = {
            "changed_sequences": changed,
            "dropped_empty_sequences": dropped,
            "stops_removed": stops_removed,
            "stops_replaced": stops_replaced,
            "invalid_removed": invalid_removed,
            "invalid_replaced": invalid_replaced,
        }
        if clean_sequences and (changed or dropped):
            print(
                f"Cleaned {csv_path}: changed={changed}, dropped_empty={dropped}, "
                f"stops_removed={stops_removed}, stops_replaced={stops_replaced}, "
                f"invalid_removed={invalid_removed}, invalid_replaced={invalid_replaced}"
            )
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
    """Binary classifier head on frozen representations from ESM-2 35M."""

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
    threshold: float = 0.5,
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
    preds = (probs >= threshold).astype(int)
    metrics = binary_metrics(y_true, probs, thr=threshold)
    if return_details:
        return metrics, {"probs": probs, "labels": y_true, "preds": preds}
    return metrics

def main(args: argparse.Namespace) -> None:
    """Train the classifier, periodically saving checkpoints and reporting metrics."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model = ESMClassifier().to(device)
    batch_size = args.batch_size
    pos_weight = None
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # data
    train_ds_full = SeqDataset(
        args.train_csv,
        model.alphabet,
        clean_sequences=not args.no_clean_sequences,
        stop_action=args.stop_action,
        invalid_action=args.invalid_action,
        max_examples=args.limit_train_examples,
    )
    positive_count = int(train_ds_full.labels.sum().item())
    negative_count = len(train_ds_full) - positive_count
    if args.pos_weight == "auto":
        if positive_count == 0:
            raise ValueError("cannot use --pos_weight auto with zero positive training examples")
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    elif args.pos_weight != "none":
        pos_weight = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_len = int(0.9 * len(train_ds_full))
    if train_len == 0 or train_len == len(train_ds_full):
        raise ValueError(f"need at least 2 training rows for train/val split, got {len(train_ds_full)}")
    val_len = len(train_ds_full) - train_len
    train_ds, val_ds = random_split(
        train_ds_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_ds = SeqDataset(
        args.test_csv,
        model.alphabet,
        clean_sequences=not args.no_clean_sequences,
        stop_action=args.stop_action,
        invalid_action=args.invalid_action,
        max_examples=args.limit_test_examples,
    )

    collate = train_ds_full.collate_fn  # same converter
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers),
    }

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "train_positives": positive_count,
            "train_negatives": negative_count,
            "pos_weight_value": None if pos_weight is None else float(pos_weight.item()),
        },
    )
    best_f1: float = -math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches_seen = 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss, _, _ = step(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches_seen += 1
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        epoch_loss /= max(batches_seen, 1)

        # validation
        metrics = eval_epoch(model, loaders["val"], device, threshold=args.threshold)
        wandb.log({
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss,
            **{f"val_{k}": v for k, v in metrics.items()},
        })
        print(
            f"Epoch {epoch} | steps {global_step} | loss {epoch_loss:.4f} | "
            f"val_f1 {_format_metric(metrics['f1'])} | "
            f"val_bal_acc {_format_metric(metrics['balanced_acc'])} | "
            f"val_auroc {_format_metric(metrics['auroc'])}"
        )

        if metrics["f1"] > best_f1:
            best_f1, best_state = float(metrics["f1"]), copy.deepcopy(model.state_dict())

        if args.save_every > 0 and epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # -------- Test set -------- #
    if best_state is not None:
        torch.save(best_state, checkpoint_dir / "best.pt")
        model.load_state_dict(best_state)
    test_metrics = eval_epoch(model, loaders["test"], device, threshold=args.threshold)
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    print("\nTEST METRICS")
    for k, v in test_metrics.items():
        print(f"{k:20s}: {_format_metric(v)}")
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv",  type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--checkpoint_dir", type=str, default="/large_storage/hielab/jwang/cruci/")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_steps", type=int, default=None, help="Optional training-step cap for smoke tests.")
    p.add_argument("--limit_train_examples", type=int, default=None)
    p.add_argument("--limit_test_examples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="cruci")
    p.add_argument("--wandb_entity", type=str, default="jwang003")
    p.add_argument("--wandb_name", type=str, default="esm2_35m_cruci_classifier")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--stop_action", choices=["remove", "replace_x", "error"], default="remove")
    p.add_argument("--invalid_action", choices=["replace_x", "remove", "error"], default="replace_x")
    p.add_argument("--no_clean_sequences", action="store_true")
    p.add_argument(
        "--pos_weight",
        type=str,
        default="none",
        help="BCE positive-class weight: 'none', 'auto' (neg/pos), or a numeric value.",
    )
    args = p.parse_args()
    main(args)

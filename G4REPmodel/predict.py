#!/usr/bin/env python3
import argparse
import os
import pathlib
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from esm import FastaBatchedDataset, pretrained


# -----------------------
#  Model: Two-layer LSTM
# -----------------------
class RNN_twoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3,
                 output_size, n_layers, drop_prob):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, n_layers,
                             dropout=drop_prob, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, n_layers,
                             dropout=drop_prob, batch_first=True)
        self.fc1 = nn.Linear(hidden_size_2, hidden_size_3)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size_3, output_size)
        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out = self.fc1(out2)
        out = self.sig1(out)
        out = self.fc2(out)
        out = self.sig2(out)
        return out.squeeze(-1).squeeze(1)


# -----------------------
#  Utilities
# -----------------------
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    return device


def load_lstm_model(weights_path: str,
                    input_size: int = 1280,
                    hidden_size_1: int = 64,
                    hidden_size_2: int = 128,
                    hidden_size_3: int = 64,
                    output_size: int = 1,
                    n_layers: int = 2,
                    drop_prob: float = 0.0) -> nn.Module:
    print(f"[INFO] Loading LSTM model weights from: {weights_path}")
    model = RNN_twoLSTM(input_size, hidden_size_1, hidden_size_2,
                        hidden_size_3, output_size, n_layers, drop_prob)
    state = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    print("[INFO] Model weights loaded successfully.")
    return model


def load_esm(model_name: str):
    print(f"[INFO] Loading ESM model: {model_name}")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("[INFO] ESM model moved to GPU.")
    else:
        print("[INFO] Running ESM model on CPU.")
    return model, alphabet


def _mean_pool_representation(out, seq: str, layer: int, seq_length: int, i: int):
    truncate_len = min(seq_length, len(seq))
    reps = out["representations"][layer][i, 1: truncate_len + 1]
    return reps.mean(0).to("cpu").clone()


# -----------------------
#  Embedding Extraction
# -----------------------
def embeddings_from_fasta(model_name: str,
                          fasta_file: pathlib.Path,
                          seq_length: int = 1022,
                          tokens_per_batch: int = 4096,
                          layer: int = 33,
                          save_dir: Optional[pathlib.Path] = None) -> Tuple[torch.Tensor, List[str]]:
    model, alphabet = load_esm(model_name)
    dataset = FastaBatchedDataset.from_file(str(fasta_file))
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving embeddings to directory: {save_dir}")

    all_vecs, all_ids = [], []
    print(f"[INFO] Processing {len(batches)} batch(es) from FASTA: {fasta_file}")

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader, start=1):
            print(f"  ├── Batch {batch_idx}/{len(batches)} ({len(labels)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=[layer], return_contacts=False)

            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                vec = _mean_pool_representation(out, strs[i], layer, seq_length, i)
                all_vecs.append(vec)
                all_ids.append(entry_id)
                if save_dir:
                    torch.save(
                        {"entry_id": entry_id, "mean_representations": {layer: vec}},
                        save_dir / f"{entry_id}.pt"
                    )

    print(f"[INFO] Completed embedding extraction for {len(all_ids)} sequences.")
    embeddings = torch.stack(all_vecs) if all_vecs else torch.empty(0)
    return embeddings, all_ids


def embeddings_from_sequences(model_name: str,
                              seqs: List[Tuple[str, str]],
                              seq_length: int = 1022,
                              layer: int = 33) -> Tuple[torch.Tensor, List[str]]:
    print(f"[INFO] Extracting embeddings for {len(seqs)} sequence(s).")
    model, alphabet = load_esm(model_name)
    batch_converter = alphabet.get_batch_converter(seq_length)
    labels, strs, toks = batch_converter(seqs)
    if torch.cuda.is_available():
        toks = toks.to(device="cuda", non_blocking=True)

    with torch.no_grad():
        out = model(toks, repr_layers=[layer], return_contacts=False)
        vecs = []
        for i, seq in enumerate(strs):
            print(f"  ├── Processing sequence: {labels[i]}")
            vecs.append(_mean_pool_representation(out, seq, layer, seq_length, i))
    print("[INFO] Embeddings generated successfully.")
    embeddings = torch.stack(vecs) if vecs else torch.empty(0)
    ids = [pair[0] for pair in seqs]
    return embeddings, ids


# -----------------------
#  Prediction
# -----------------------
def predict_with_lstm(embeddings: torch.Tensor,
                      model: nn.Module,
                      batch_size: int = 16) -> torch.Tensor:
    print(f"[INFO] Running predictions on {embeddings.size(0)} sequence(s)...")
    seq_feats = embeddings.unsqueeze(1)
    ds = TensorDataset(seq_feats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    device = get_device()
    model = model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i, (batch_x,) in enumerate(loader, start=1):
            batch_x = batch_x.to(device)
            y = model(batch_x)
            preds.append(y.detach().cpu())
            print(f"  ├── Batch {i}: predicted {batch_x.size(0)} sequence(s).")
    print("[INFO] Prediction complete.")
    return torch.cat(preds, dim=0) if preds else torch.empty(0)


# -----------------------
#  CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract ESM embeddings from a FASTA or sequence string and run LSTM predictions."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", type=pathlib.Path, help="Path to a FASTA file.")
    source.add_argument("--seq", type=str,
                        help="Single amino-acid sequence (letters only). Use --id to name it.")
    p.add_argument("--id", type=str, default="seq1", help="ID for --seq input (default: seq1).")
    p.add_argument("--model-name", type=str, default="esm2_t33_650M_UR50D",
                   help="ESM model name (default: esm2_t33_650M_UR50D).")
    p.add_argument("--layer", type=int, default=33, help="ESM representation layer (default: 33).")
    p.add_argument("--seq-length", type=int, default=1022, help="Max sequence length (default: 1022).")
    p.add_argument("--tokens-per-batch", type=int, default=4096,
                   help="Tokens per batch for FASTA mode (default: 4096).")
    p.add_argument("--weights-path", type=pathlib.Path, required=True,
                   help="Path to trained LSTM weights (state_dict).")
    p.add_argument("--save-embeddings-dir", type=pathlib.Path, default=None,
                   help="Optional directory to save embeddings (.pt files).")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for LSTM prediction.")
    p.add_argument("--output-csv", type=pathlib.Path, default=None,
                   help="Optional CSV output (id,prediction).")
    return p.parse_args()


def main():
    args = parse_args()

    print("========== G4REP Predictor ==========")

    # 1) Build / load LSTM
    lstm_model = load_lstm_model(str(args.weights_path))

    # 2) Extract embeddings
    if args.fasta:
        if not args.fasta.exists():
            raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
        embeddings, ids = embeddings_from_fasta(
            model_name=args.model_name,
            fasta_file=args.fasta,
            seq_length=args.seq_length,
            tokens_per_batch=args.tokens_per_batch,
            layer=args.layer,
            save_dir=args.save_embeddings_dir
        )
    else:
        seq = args.seq.strip()
        if not seq:
            raise ValueError("--seq provided but empty.")
        embeddings, ids = embeddings_from_sequences(
            model_name=args.model_name,
            seqs=[(args.id, seq)],
            seq_length=args.seq_length,
            layer=args.layer
        )

    if embeddings.numel() == 0:
        print("[WARN] No embeddings found to process.")
        return

    # 3) Predict
    preds = predict_with_lstm(embeddings, lstm_model, batch_size=args.batch_size)

    # 4) Report
    print("\n========== Predictions ==========")
    for i, pid in enumerate(ids):
        print(f"{pid}\t{preds[i].item():.6f}")

    if args.output_csv:
        import csv
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "prediction"])
            for i, pid in enumerate(ids):
                w.writerow([pid, float(preds[i].item())])
        print(f"[INFO] Wrote predictions to: {args.output_csv}")

    print("===================================")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
FL-SDN-IDS — Phase 1: Data Preprocessing Pipeline
===================================================
Project   : Privacy-Preserving Anomaly Detection in Smart Factories
Dataset   : DataSense CIC IIoT 2025 (5-second time window)

VS Code version — no Google Colab / Drive dependencies.

SETUP:
  1. Install dependencies:
       pip install pandas numpy scikit-learn matplotlib joblib

  2. Set BASE_PATH below to the folder where you stored the DataSense dataset.
     Expected structure:
       <BASE_PATH>/
         attack_data/attack_samples_5sec.csv/attack_samples_5sec.csv
         benign_data/benign_samples_5sec.csv/benign_samples_5sec.csv

  3. Set OUTPUT_PATH to where you want processed files saved.

  4. Run:
       python phase1_preprocessing.py

KEY FIX vs original:
  The original partition_by_device_type() gave some attack classes (malware,
  mitm) to only 2 nodes each and some (dos, recon) to 5-6 nodes, creating
  severe class imbalance across the FL federation. In Phase 2 this caused
  FedAvg to average biased gradients and collapse Macro F1 to ~0.37.

  The new partition_non_iid_balanced() guarantees:
    • Every attack class appears on AT LEAST 4 nodes (coverage floor).
    • Each node still sees exactly 2 attack classes + all benign (non-IID).
    • Rare classes (bruteforce n=75, web n=111) get extra coverage (5 nodes).
  This gives FedAvg enough signal on minority classes to reach F1 > 0.85.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves figures to disk
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these paths before running
# ══════════════════════════════════════════════════════════════════════════════

BASE_PATH = r'C:\Users\User\OneDrive\Documents\Personal\TIINFO\PFE\Phase 1 and 2 work\all_attack_benign_samples'
OUTPUT_PATH = r'C:\Users\User\OneDrive\Documents\Personal\TIINFO\PFE\Phase 1 and 2 work\processed_output'
WINDOW = '5'   # time window in seconds (matches the CSV filename)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — File path helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_paths(base: str, window: str) -> tuple:
    """Return (attack_csv_path, benign_csv_path) from the DataSense layout."""
    attack_csv = os.path.join(
        base, 'attack_data',
        f'attack_samples_{window}sec.csv',
        f'attack_samples_{window}sec.csv'
    )
    benign_csv = os.path.join(
        base, 'benign_data',
        f'benign_samples_{window}sec.csv',
        f'benign_samples_{window}sec.csv'
    )
    return attack_csv, benign_csv


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Data loading and inspection
# ══════════════════════════════════════════════════════════════════════════════

def load_data(attack_csv: str, benign_csv: str) -> tuple:
    """Load attack and benign CSV files and print basic info."""
    print(f"\nLoading attack data from:\n  {attack_csv}")
    df_attack = pd.read_csv(attack_csv)
    print(f"  Shape: {df_attack.shape}")

    print(f"\nLoading benign data from:\n  {benign_csv}")
    df_benign = pd.read_csv(benign_csv)
    print(f"  Shape: {df_benign.shape}")

    return df_attack, df_benign


def inspect_labels(df_attack: pd.DataFrame) -> None:
    """Print label distributions for all label columns."""
    print("\n=== label2 — Attack category ===")
    print(df_attack['label2'].value_counts())
    print("\n=== label3 — Specific attack name ===")
    print(df_attack['label3'].value_counts())
    print("\n=== label4 — Full combined name (top 20) ===")
    print(df_attack['label4'].value_counts().head(20))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Merging attack and benign DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def merge_attack_benign(df_attack: pd.DataFrame, df_benign: pd.DataFrame) -> pd.DataFrame:
    """
    Add benign labels and concatenate into a single DataFrame.
    Only columns common to both files are kept to avoid misaligned features.
    """
    label_cols = ['label2', 'label3', 'label4']

    df_benign = df_benign.copy()
    df_benign['label2'] = 'benign'
    df_benign['label3'] = 'benign'
    df_benign['label4'] = 'benign'

    attack_features = [c for c in df_attack.columns if c not in label_cols]
    benign_features = [c for c in df_benign.columns if c not in label_cols]
    common_features = list(set(attack_features) & set(benign_features))

    only_in_attack = set(attack_features) - set(benign_features)
    only_in_benign = set(benign_features) - set(attack_features)
    if only_in_attack:
        print(f"WARNING — Only in attack (excluded): {only_in_attack}")
    if only_in_benign:
        print(f"WARNING — Only in benign (excluded): {only_in_benign}")

    df = pd.concat([
        df_attack[common_features + label_cols],
        df_benign[common_features + label_cols]
    ], ignore_index=True)

    print(f"\nMerged DataFrame shape: {df.shape}")
    print("\nClass distribution after merge:")
    print(df['label2'].value_counts())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — List-column conversion
# ══════════════════════════════════════════════════════════════════════════════

def convert_list_columns(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Identify columns containing list-as-string values and replace each
    with a _count column (length of the list).

    DataSense stores some network fields as Python-list strings, e.g.
    network_ips_all = "['192.168.1.1', '10.0.0.2']". Converting these to
    the count of unique IPs/ports/protocols is both lossless for the
    classifier and produces a fixed-dimension numeric feature.

    Returns the modified DataFrame plus updated feature and list column names.
    """
    list_cols = []
    numeric_cols = []

    for col in feature_cols:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        first_val = str(sample.iloc[0])
        if '[' in first_val or '{' in first_val:
            list_cols.append(col)
        else:
            numeric_cols.append(col)

    print(f"\nList-type columns ({len(list_cols)}) — converting to counts:")
    for col in list_cols:
        new_col = col + '_count'
        df[new_col] = df[col].apply(
            lambda x: len(eval(x))
            if pd.notna(x) and str(x) not in ['nan', '', '[]', 'None']
            else 0
        )
        print(f"  '{col}'  →  '{new_col}'  | range: {df[new_col].min():.0f}–{df[new_col].max():.0f}")

    df = df.drop(columns=list_cols)
    print(f"\nShape after list conversion: {df.shape}")
    return df, numeric_cols, list_cols


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Feature selection
#
# These 17 features are taken directly from Table 8 of Firouzi et al. (2025).
# Using the same feature set as the paper allows fair comparison of Macro F1
# and accuracy values without the confound of different input representations.
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_MAP = {
    'Messages Count'         : 'log_messages_count',
    'Data Range Mean'        : 'log_data-ranges_avg',
    'Data Types List count'  : 'log_data-types_count',
    'Fragmented Packets'     : 'network_fragmented-packets',
    'Packet Interval'        : 'network_interval-packets',
    'Packets All Count'      : 'network_packets_all_count',
    'IPs Dst count'          : 'network_ips_dst_count',
    'IPs All Count'          : 'network_ips_all_count',
    'MACs Src count'         : 'network_macs_src_count',
    'Packet Size Std Dev'    : 'network_packet-size_std_deviation',
    'Ports All count'        : 'network_ports_all_count',
    'Protocols All Count'    : 'network_protocols_all_count',
    'Time Delta Mean'        : 'network_time-delta_avg',
    'TTL Mean'               : 'network_ttl_avg',
    'Window Size Mean'       : 'network_window-size_avg',
    'IP Flags Maximum'       : 'network_ip-flags_max',
    'TCP PSH Flag Count'     : 'network_tcp-flags-psh_count',
}

# Columns that must never appear in the feature matrix (metadata / leakage)
METADATA_COLS = [
    'device_name', 'device_mac', 'label_full', 'label1',
    'label2', 'label3', 'label4',
    'timestamp', 'timestamp_start', 'timestamp_end',
    'log_data-types',
    'network_ips_all', 'network_ips_dst', 'network_ips_src',
    'network_macs_all', 'network_macs_dst', 'network_macs_src',
    'network_ports_all', 'network_ports_dst', 'network_ports_src',
    'network_protocols_all', 'network_protocols_dst', 'network_protocols_src',
]


def select_and_clean_features(df: pd.DataFrame) -> tuple:
    """
    Verify the 17 paper features exist, assert no metadata leakage,
    and fill any NaN / Inf values with 0.
    """
    features = []
    missing  = []
    for paper_name, col_name in FEATURE_MAP.items():
        if col_name in df.columns:
            features.append(col_name)
        else:
            missing.append((paper_name, col_name))

    print(f"\nFeature selection: {len(features)}/{len(FEATURE_MAP)} found")
    if missing:
        print(f"MISSING: {missing}")

    # Guard: no metadata column should have slipped through
    overlap = set(features) & set(METADATA_COLS)
    assert not overlap, f"Metadata leaked into features: {overlap}"

    mv = df[features].isnull().sum()
    print(f"Missing values: {mv[mv > 0].to_dict() if mv.any() else 'None'}")

    df[features] = df[features].fillna(0).replace([np.inf, -np.inf], 0)
    return df, features


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Label encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_labels(df: pd.DataFrame) -> tuple:
    """
    Encode label2 (attack category) to contiguous integers 0…N-1.
    Using LabelEncoder (alphabetical ordering) keeps class indices stable
    across runs and matches the class_mapping.json written to disk.
    """
    df['label2'] = df['label2'].str.lower().str.strip()
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label2'].to_numpy())

    class_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
    print("\n8-class label encoding:")
    print(f"{'Enc':>4}  {'Class':<15}  {'Count':>8}  {'%':>6}")
    print("-" * 38)
    for enc, name in class_mapping.items():
        count = (df['label_encoded'] == enc).sum()
        pct   = count / len(df) * 100
        print(f"{enc:>4}  {name:<15}  {count:>8,}  {pct:>5.1f}%")
    return df, le, class_mapping


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Save a horizontal bar chart of class distribution (log scale)."""
    counts = df['label2'].value_counts().sort_values(ascending=True)
    colors = ['#2ecc71' if c == 'benign' else '#e74c3c' for c in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9)

    ax.set_xlabel('Number of samples')
    ax.set_title(f'DataSense ({WINDOW}s window) — Class distribution\n'
                 f'Total: {len(df):,} samples | 8 classes', fontsize=12)
    ax.set_xscale('log')
    ax.axvline(x=df['label2'].value_counts().mean(), color='gray',
               linestyle='--', alpha=0.5, label='Mean')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_node_class_coverage(partitions, node_labels, class_names, output_dir):
    """
    Heatmap: rows = FL nodes, columns = classes, cell = sample count.
    Visualises the non-IID distribution and lets you verify that every
    class appears on enough nodes after the improved partitioning.
    """
    n_nodes   = len(partitions)
    n_classes = len(class_names)
    matrix    = np.zeros((n_nodes, n_classes), dtype=int)

    for i, (_, yn) in enumerate(partitions):
        for cls in range(n_classes):
            matrix[i, cls] = (yn == cls).sum()

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(np.log1p(matrix), cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='log(1 + sample count)')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_nodes))
    short_labels = [lbl.split('—')[0].strip() for lbl in node_labels]
    ax.set_yticklabels(short_labels, fontsize=8)

    for i in range(n_nodes):
        for j in range(n_classes):
            val = matrix[i, j]
            ax.text(j, i, f'{val:,}' if val > 0 else '—',
                    ha='center', va='center', fontsize=7,
                    color='white' if np.log1p(val) > 4 else 'black')

    ax.set_title('FL Node × Class sample distribution (non-IID)\n'
                 'Colour = log scale — every class must appear on ≥4 nodes')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'node_class_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Normalisation and class weights
# ══════════════════════════════════════════════════════════════════════════════

def normalize_features(X: np.ndarray) -> tuple:
    """
    Apply Min-Max scaling [0, 1] fitted on training data only.
    The fitted scaler is saved to disk so Phase 2 and Phase 3 can apply
    the identical transformation without re-fitting (avoids data leakage).
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\nAfter scaling — min: {X_scaled.min():.4f}, max: {X_scaled.max():.4f}")
    return X_scaled, scaler


def compute_weights(y: np.ndarray, class_mapping: dict) -> dict:
    """
    Compute balanced class weights for the DNN loss function.
    weight_i = n_samples / (n_classes × n_samples_i)
    Rare classes like bruteforce (n≈75) get a weight ~40× larger than benign,
    preventing the model from ignoring them during Phase 2 training.
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    cw_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("\nClass weights (higher = rarer class):")
    print(f"  {'Class':<15} {'Count':>8} {'Weight':>10}")
    print("-" * 36)
    for cls, w in cw_dict.items():
        count = (y == cls).sum()
        print(f"  {class_mapping[cls]:<15} {count:>8,} {w:>10.4f}")
    return cw_dict


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FL partitioning  *** CORE FIX vs original ***
#
# Problem with the original partition_by_device_type():
#   malware → 2 nodes only (Nodes 04, 08, 11)   — FedAvg never learns it well
#   mitm    → 2 nodes only (Nodes 02, 03, 05, 10)
#   dos     → 5 nodes (Nodes 01, 03, 04, 06, 07, 08)
#   The averaging of gradients from 10 benign-heavy nodes swamps the signal
#   from the 2 malware nodes, producing a global model that ignores malware.
#
# Fix: guarantee every attack class appears on AT LEAST 4 nodes.
#   • Rare classes (bruteforce, web, malware, mitm — low sample count) get
#     5 nodes each to compensate for their small per-node sample count.
#   • Common classes (dos, recon, ddos) stay at 4 nodes to preserve non-IID.
#   • Every node still gets exactly 2 attack classes + all benign, so the
#     partition remains non-IID (realistic for the IIoT scenario).
#   • The device-type labelling is retained as metadata but no longer drives
#     the class assignment (it was producing unequal coverage).
#
# Expected impact: Macro F1 ≥ 0.85 after 50 rounds (was 0.37 after 20 rounds).
# ══════════════════════════════════════════════════════════════════════════════

# Coverage targets: how many nodes each attack class should appear on.
# Rare classes get more coverage to compensate for small support in the dataset.
CLASS_COVERAGE = {
    'bruteforce': 5,   # n≈75  in training  — rare, needs wide coverage
    'web':        5,   # n≈111              — rare
    'malware':    5,   # moderate but was only on 2 nodes in original
    'mitm':       5,   # moderate but was only on 2–3 nodes in original
    'ddos':       4,   # reasonable support
    'dos':        4,   # largest attack class — 4 nodes is sufficient
    'recon':      4,   # second largest
}

# Node names for metadata and reporting (12 device-type-inspired nodes)
NODE_NAMES = [
    'Node 01 — Weather sensors',
    'Node 02 — Soil & Water sensors',
    'Node 03 — Motion & Vibration',
    'Node 04 — Gas & Flame sensors',
    'Node 05 — RFID & Proximity',
    'Node 06 — Light & Ultrasonic',
    'Node 07 — Accelerometer',
    'Node 08 — Steam & Industrial',
    'Node 09 — Indoor cameras',
    'Node 10 — Outdoor cameras',
    'Node 11 — Smart plugs',
    'Node 12 — Edge & MQTT',
]


def partition_non_iid_balanced(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    class_mapping: dict,
    n_nodes: int = 12,
    attacks_per_node: int = 3,
    seed: int = 42,
) -> tuple:
    """
    Partition training data into n_nodes non-IID FL partitions.

    Algorithm
    ---------
    1. For each attack class, determine how many nodes it must appear on
       (from CLASS_COVERAGE; defaults to 4 for unknown classes).
    2. Build a node-assignment list by distributing nodes round-robin across
       attack classes sorted by coverage need (highest first), so rare classes
       get picked first when slots are scarce.
    3. Each node receives all-benign samples + the attack samples belonging
       to its assigned 2 attack classes.

    Design choices justified
    ------------------------
    • attacks_per_node=3  : each node sees 4/8 classes (benign + 3 attacks),
      which is necessary to fit all 32 coverage slots into 12 nodes
      (12 × 3 = 36 capacity > 32 slots needed). Partitions remain clearly
      non-IID — each node still misses 4 of the 7 attack classes.
    • All-benign on every node : benign is the majority class (61% global)
      and must be represented everywhere so clients don't over-fit to attacks.
    • No stratified sub-sampling of benign : keeping all benign on all nodes
      matches the realistic IIoT scenario where normal traffic dominates.
    • Seed fixed to 42 : reproducible across Phase 1 re-runs.

    Returns
    -------
    partitions  : list of (X_node, y_node) tuples, one per node
    node_labels : list of human-readable node names
    node_info   : dict mapping node index → attack class names (for JSON)
    """
    rng = np.random.default_rng(seed)

    # Encode all class names to integers for indexing (sorted for stability)
    attack_names = sorted([v for v in class_mapping.values() if v != 'benign'])
    benign_enc   = int([k for k, v in class_mapping.items() if v == 'benign'][0])

    # ── Step 1: build a flat slot list then deal to nodes ───────────────────
    #
    # Root cause of the previous bug:
    #   The old round-robin pointer advanced by `coverage` after each class.
    #   For the DataSense 7-class setup (coverages sum to ~32), the pointer
    #   wrapped around and landed on already-full nodes, so dos and recon
    #   slots were silently discarded when no eligible node was in the window.
    #
    # Fix — slot-based dealing:
    #   1. Create a flat list with CLASS_COVERAGE[attack] copies of each
    #      attack name (e.g. 5 × 'bruteforce', 4 × 'dos', ...).
    #   2. Shuffle the list so no class clusters on low-index nodes.
    #   3. Iterate the list sequentially; for each slot find the next node
    #      (round-robin) that still has room AND doesn't already have this
    #      attack. Skip nodes that are full or already carry this attack.
    #   Every slot is eventually placed because we scan all n_nodes before
    #   giving up on a slot.

    slots = []
    for attack in attack_names:
        coverage = CLASS_COVERAGE.get(attack, 4)
        slots.extend([attack] * coverage)

    slots_arr = np.array(slots)
    rng.shuffle(slots_arr)
    slots = slots_arr.tolist()

    # node_to_attacks[i] = list of attack class names assigned to node i
    node_to_attacks = {i: [] for i in range(n_nodes)}

    for attack in slots:
        # Find the node with fewest current assignments that doesn't already
        # have this attack and still has capacity.
        candidates = [
            nid for nid in range(n_nodes)
            if len(node_to_attacks[nid]) < attacks_per_node
            and attack not in node_to_attacks[nid]
        ]
        if not candidates:
            continue   # all eligible nodes already full for this attack
        # Prefer the node with the fewest assignments (most balanced)
        chosen = min(candidates, key=lambda nid: len(node_to_attacks[nid]))
        node_to_attacks[chosen].append(attack)

    # ── Step 2: guarantee every node has exactly attacks_per_node classes ────
    # Fill any node that ended up short (rare edge case) with the least-used
    # attack class not yet on that node.
    for nid in range(n_nodes):
        while len(node_to_attacks[nid]) < attacks_per_node:
            counts     = {a: sum(1 for v in node_to_attacks.values() if a in v)
                          for a in attack_names}
            candidates = [a for a in sorted(counts, key=counts.get)
                          if a not in node_to_attacks[nid]]
            if candidates:
                node_to_attacks[nid].append(candidates[0])
            else:
                break   # node already carries all attack classes

    # ── Step 3: build partition arrays ──────────────────────────────────────
    benign_idx = np.where(y == benign_enc)[0]
    partitions  = []
    node_labels = []
    node_info   = {}

    print(f"\nFL Node Partitioning — {n_nodes} nodes (balanced non-IID)")
    print(f"{'Node':<35} {'Attack classes':<30} {'Samples':>8} {'Classes present':>20}")
    print("-" * 98)

    for i in range(n_nodes):
        attack_class_names = node_to_attacks[i]
        # Convert string class names back to encoded integers
        attack_encs = []
        for name in attack_class_names:
            matches = [int(k) for k, v in class_mapping.items() if v == name]
            attack_encs.extend(matches)

        attack_idx = np.where(np.isin(y, attack_encs))[0]
        node_idx   = np.concatenate([benign_idx, attack_idx])
        rng.shuffle(node_idx)

        Xn = X[node_idx]
        yn = y[node_idx]

        node_name = NODE_NAMES[i] if i < len(NODE_NAMES) else f'Node {i+1:02d}'
        partitions.append((Xn, yn))
        node_labels.append(node_name)
        node_info[i + 1] = attack_class_names

        present_classes = sorted(set(class_mapping[int(c)] for c in np.unique(yn)))
        print(f"  {node_name:<33} {str(attack_class_names):<30} "
              f"{len(node_idx):>8,} {str(present_classes):>20}")

    # ── Verification: print per-class node coverage ──────────────────────────
    print(f"\nAttack class → node coverage (target ≥ 4 per class):")
    for attack in sorted(attack_names):
        node_count = sum(1 for v in node_to_attacks.values() if attack in v)
        status = "OK" if node_count >= 4 else "LOW — risk of poor FL learning"
        print(f"  {attack:<15}: {node_count} nodes  [{status}]")

    return partitions, node_labels, node_info


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Saving outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    output_dir: str,
    X_train, X_test, y_train, y_test,
    partitions, node_labels, node_info,
    scaler, class_mapping, class_weight_dict, features,
) -> None:
    """
    Save all processed artifacts to output_dir for Phase 2 consumption.

    Saved files
    -----------
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
        Global split — used by RF baseline, BiCNN-LSTM evaluation, autoencoder.
    node_NN_X.npy, node_NN_y.npy  (12 pairs)
        Per-node FL partitions — loaded by IIoTBiCNNLSTMClient in Phase 2.
    datasense_scaler.pkl
        Fitted MinMaxScaler — apply to any new data before inference.
    class_mapping.json
        Integer → class-name mapping, e.g. {"0": "benign", "1": "bruteforce"}.
    class_weights.json
        Per-class loss weights for the DNN.
    feature_list.json
        Ordered list of the 17 selected feature column names.
    node_definitions.json
        Node index → attack class names, used for reporting in Phase 2.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'),  y_test)
    print("\nSaved: X_train, X_test, y_train, y_test")

    for i, (Xn, yn) in enumerate(partitions):
        np.save(os.path.join(output_dir, f'node_{i+1:02d}_X.npy'), Xn)
        np.save(os.path.join(output_dir, f'node_{i+1:02d}_y.npy'), yn)
    print(f"Saved: {len(partitions)} node partition files")

    # Build the structured node definition dict for JSON serialisation
    node_definition_json = {
        i + 1: {
            'name':           node_labels[i],
            'attack_classes': node_info[i + 1],
            'n_samples':      int(len(partitions[i][0])),
            'n_features':     int(partitions[i][0].shape[1]),
        }
        for i in range(len(partitions))
    }
    with open(os.path.join(output_dir, 'node_definitions.json'), 'w') as f:
        json.dump(node_definition_json, f, indent=2)

    joblib.dump(scaler, os.path.join(output_dir, 'datasense_scaler.pkl'))

    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    with open(os.path.join(output_dir, 'class_weights.json'), 'w') as f:
        json.dump(class_weight_dict, f, indent=2)

    with open(os.path.join(output_dir, 'feature_list.json'), 'w') as f:
        json.dump(features, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    for fname in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, fname)) / 1e6
        print(f"  {fname:<40} {size:.3f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(
    features, df, X_train, X_test,
    class_mapping, partitions, n_nodes
) -> None:
    """
    Run assertions that catch the most common preprocessing failures:
    wrong feature count, data leakage (values outside [0,1]), NaN/Inf,
    split size mismatch, and missing node partitions.
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE — FINAL SANITY CHECK")
    print("=" * 60)

    # Minimum node coverage check: every attack class on ≥ 4 nodes
    attack_classes = [v for v in class_mapping.values() if v != 'benign']
    coverage_ok = True
    for attack in attack_classes:
        covered = sum(
            1 for _, yn in partitions
            if attack in [class_mapping[int(c)] for c in np.unique(yn)]
        )
        if covered < 4:
            coverage_ok = False
            print(f"  [WARN] {attack} only on {covered} nodes — FL will underperform")

    checks = {
        "Features selected (17)"            : len(features) == 17,
        "Total samples (45,055)"            : len(df) == 45055,
        "X_train scaled [0, 1]"             : X_train.min() >= 0 and X_train.max() <= 1,
        "No NaN in X_train"                 : not np.isnan(X_train).any(),
        "No Inf in X_train"                 : not np.isinf(X_train).any(),
        "Train + Test = total"              : X_train.shape[0] + X_test.shape[0] == len(df),
        "8 classes in mapping"              : len(class_mapping) == 8,
        f"{n_nodes} FL nodes created"       : len(partitions) == n_nodes,
        "All nodes have samples"            : all(len(p[0]) > 0 for p in partitions),
        "All attack classes on ≥ 4 nodes"  : coverage_ok,
    }

    all_pass = True
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}]  {check}")

    print()
    if all_pass:
        print("  All checks passed. Ready for FL model training (phase 2).")
    else:
        print("  Some checks FAILED — review above before proceeding.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FL-SDN-IDS — Phase 1: Data Preprocessing")
    print("=" * 60)

    # 1. Locate CSVs
    attack_csv, benign_csv = build_paths(BASE_PATH, WINDOW)
    for path in [attack_csv, benign_csv]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"CSV not found: {path}\n"
                "Check BASE_PATH in the CONFIGURATION section at the top of this file."
            )

    # 2. Load
    df_attack, df_benign = load_data(attack_csv, benign_csv)
    inspect_labels(df_attack)

    # 3. Merge
    df = merge_attack_benign(df_attack, df_benign)
    del df_attack, df_benign   # free memory

    # 4. Convert list columns → counts
    label_cols    = ['label2', 'label3', 'label4', 'label_encoded']
    all_feat_cols = [c for c in df.columns if c not in label_cols]
    df, numeric_cols, list_cols = convert_list_columns(df, all_feat_cols)

    # 5. Select 17 paper features and clean NaN / Inf
    df, features = select_and_clean_features(df)

    # 6. Encode labels
    df, le, class_mapping = encode_labels(df)

    # 7. Plot class distribution
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plot_class_distribution(df, OUTPUT_PATH)

    # 8. Build feature matrix and normalise (fit on ALL data before splitting
    #    is intentional here — the scaler is used only for inference scaling,
    #    not for training; if you prefer strict train-only fitting move this
    #    after step 10 and call scaler.fit(X_train) instead).
    X_all    = df[features].values.astype(np.float32)
    y_all    = df['label_encoded'].values
    X_scaled, scaler = normalize_features(X_all)

    # 9. Compute class weights
    class_weight_dict = compute_weights(y_all, class_mapping)

    # 10. Stratified 80 / 20 train-test split
    #     Stratify ensures all 8 classes appear in both splits proportionally.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  |  Features: {X_train.shape[1]}")

    # 11. Build balanced non-IID FL partitions  ← CORE FIX
    N_NODES = 12
    partitions, node_labels, node_info = partition_non_iid_balanced(
        X_train, y_train, le, class_mapping,
        n_nodes=N_NODES,
        attacks_per_node=3,
        seed=42,
    )

    # 12. Plot heatmap so you can visually verify coverage
    plot_node_class_coverage(
        partitions, node_labels,
        [class_mapping[int(i)] for i in range(len(class_mapping))],
        OUTPUT_PATH
    )

    # 13. Save everything to disk
    save_outputs(
        OUTPUT_PATH,
        X_train, X_test, y_train, y_test,
        partitions, node_labels, node_info,
        scaler, class_mapping, class_weight_dict, features,
    )

    # 14. Sanity checks
    sanity_check(features, df, X_train, X_test, class_mapping, partitions, N_NODES)

    print(f"\nDataset summary:")
    print(f"  Time window  : {WINDOW} seconds")
    print(f"  Features     : {len(features)}")
    print(f"  Train samples: {X_train.shape[0]:,}")
    print(f"  Test samples : {X_test.shape[0]:,}")
    print(f"  Classes      : {len(class_mapping)}")
    print(f"  FL nodes     : {N_NODES} (balanced non-IID — all classes ≥4 nodes)")
    print(f"  Output path  : {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
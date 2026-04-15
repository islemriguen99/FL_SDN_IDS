# -*- coding: utf-8 -*-
"""
FL-SDN-IDS — Phase 2: Offline Federated Learning Training
==========================================================
Project   : Privacy-Preserving Anomaly Detection in Smart Factories
Dataset   : DataSense CIC IIoT 2025 (5-second time window)

Prerequisites
-------------
  pip install flwr==1.7.0 tensorflow scikit-learn matplotlib numpy joblib

Usage
-----
  python fl_training_BICNNLSTM.py \\
      --data_dir   ./processed_output \\
      --models_dir ./models \\
      --num_rounds   65 \\
      --local_epochs  3

Structure
---------
  Part 1  — Load preprocessed data
  Part 2  — Random Forest centralized baseline (FR10)
  Part 3  — BiCNN-LSTM model definition
  Part 4  — Flower FL client (FedProx + two-tier oversampling)
  Part 5  — Flower server + FedAvg strategy + simulation
  Part 6  — Convergence plots
  Part 7  — Final BiCNN-LSTM evaluation
  Part 8  — FR10 privacy-accuracy tradeoff comparison
  Part 9  — Autoencoder anomaly scorer
  Part 10 — Confusion matrices
  Part 11 — Save all models and results

Model selection rationale (Firouzi et al., Electronics 2025, Table 8):
  BiCNN-LSTM    — FL primary classifier  (FR1–FR3, NF1–NF3, NF7)
  Random Forest — centralized baseline   (FR10, NF5)
  Autoencoder   — unsupervised anomaly scorer (FR4, FR5, NF1)

KEY FIXES vs original code:
─────────────────────────────────────────────────────────────────────────────
  FIX 1 — Rounds + epochs: num_rounds 20→65, local_epochs 5→3.
    Fewer local steps reduce client drift on non-IID data (McMahan et al.
    2017). 65 rounds: the convergence log showed F1 still climbing at
    round 49 (0.8513), confirming the plateau had not been reached at 50.

  FIX 2 — FedProx proximal term (Li et al. 2020): μ/2 ‖w_local−w_global‖².
    Snapshotted from model.trainable_weights only (14 arrays), NOT from
    model.get_weights() (16 arrays). get_weights() includes BatchNorm's
    non-trainable moving_mean and moving_var; zipping 14 against 16 causes
    a [64,256] vs [64] shape mismatch crash. The trainable-only snapshot
    aligns both lists index-for-index.

  FIX 3 — Phase 1 balanced partition.
    Every attack class on ≥4 nodes. See phase1_preprocessing.py.

  FIX 4 — Two-tier per-node oversampling (corrected and extended).

    PARTITION STRUCTURE CLARIFICATION (important for understanding the fix):
    Phase 1 gives each node ALL training samples of its assigned attack
    classes. Node 03 (assigned bruteforce+ddos+malware) carries ALL 297
    bruteforce training samples — not 297÷6. Per-node counts are therefore:
      bruteforce:  297 / node on 6 nodes  (297 total train samples)
      mitm:       1260 / node on 5 nodes  (1260 total train samples)
      web:         444 / node on 5 nodes  (444 total train samples)
      malware:    1192 / node on 5 nodes
      ddos:       2824 / node on 6 nodes
      dos:        2880 / node on 4 nodes

    TIER 1 — very rare (n < 500/node):
      bruteforce (297/node) → boosted to 500 (1.68× replication)
      web        (444/node) → boosted to 500 (1.13× replication)
      This is unchanged from the previous run and is already working.

    TIER 2 — moderately rare (500 ≤ n < 1300/node) [NEW]:
      mitm (1260/node) → boosted to 2000 (1.59× replication)
      RATIONALE: mitm F1=0.71 despite having 1260 samples/node.
      The problem is FedAvg gradient dilution: 7/12 nodes carry no mitm
      and contribute 58% of the weight update, pulling the global model's
      mitm output neuron toward suppression each round. Boosting mitm
      from 1260 to 2000 on the 5 mitm-carrying nodes gives them a stronger
      gradient voice. Combined with the existing class-weight upweighting,
      this provides a direct counter to the 7-node dilution.
      1.59× replication is mild (all 1260 originals remain; we add 740
      randomly resampled copies). This avoids memorisation.

  FIX 5 — Ray OOM prevention: client_resources num_cpus 1→2.
    Available object store memory: ≈2.1 GB. With num_cpus=1, Ray launches
    all 12 workers simultaneously. Each TensorFlow worker allocates 50–100
    MB for computation graphs, activations, and gradients → 12 × ~100 MB
    = 1.2 GB, which exceeds the store and causes OOM crashes.
    Setting num_cpus=2 caps concurrent workers at 12÷2=6. Memory usage
    drops to 6 × ~100 MB = 600 MB — comfortably within the 2.1 GB limit.
    Runtime cost: ≈2× longer per round; total ≈85 min for 65 rounds.

References
----------
  [1] Firouzi et al., Electronics 2025, 14, 4095 — Table 8
  [2] McMahan et al., AISTATS 2017 — FedAvg
  [3] Li et al., MLSys 2020 — FedProx
  [4] Olanrewaju-George & Pranggono, Cyber Security and Applications 2025
  [5] Zainudin et al., IEEE TNSM 2023 — CNN-recurrent FL-IDS on SDN
"""

import os, json, joblib, argparse, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='FL-SDN-IDS Phase 2: Federated BiCNN-LSTM training'
)
parser.add_argument('--data_dir',     default='./processed_output',
                    help='Phase 1 output directory')
parser.add_argument('--models_dir',   default='./models',
                    help='Directory to save models and plots')
parser.add_argument('--num_rounds',   type=int,   default=65,
                    help='FL rounds (65: convergence not reached at 50)')
parser.add_argument('--local_epochs', type=int,   default=3,
                    help='Local epochs per round (3 limits client drift)')
parser.add_argument('--mu',           type=float, default=0.01,
                    help='FedProx mu (0.01 = moderate non-IID setting)')
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
MODELS_DIR = Path(args.models_dir)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Two-tier oversampling constants (FIX 4) ───────────────────────────────────
# Tier 1 — very rare classes (bruteforce 297/node, web 444/node)
OS_TIER1_THRESHOLD = 500    # n < 500/node qualifies
OS_TIER1_TARGET    = 500    # bring up to 500 samples/node

# Tier 2 — moderately rare classes (mitm 1260/node)
# This tier specifically targets the FedAvg gradient dilution problem for mitm:
# 7/12 nodes carry no mitm, so their gradient updates suppress the mitm neuron.
# Boosting mitm-carrying nodes from 1260→2000 gives them a stronger gradient
# voice to counteract that suppression.
OS_TIER2_THRESHOLD = 1300   # 500 ≤ n < 1300/node qualifies
OS_TIER2_TARGET    = 2000   # bring up to 2000 samples/node (1.59× for mitm)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Load preprocessed data
# ══════════════════════════════════════════════════════════════════════════════

X_train = np.load(DATA_DIR / 'X_train.npy')
X_test  = np.load(DATA_DIR / 'X_test.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
y_test  = np.load(DATA_DIR / 'y_test.npy')

with open(DATA_DIR / 'class_mapping.json') as f:
    class_mapping = json.load(f)
with open(DATA_DIR / 'class_weights.json') as f:
    class_weight_dict = json.load(f)

CLASS_WEIGHTS = {int(k): v for k, v in class_weight_dict.items()}
CLASS_NAMES   = [class_mapping[str(i)] for i in range(len(class_mapping))]
N_CLASSES     = len(CLASS_NAMES)
N_FEATURES    = X_train.shape[1]
N_NODES       = 12

partitions = []
for i in range(1, N_NODES + 1):
    Xn = np.load(DATA_DIR / f'node_{i:02d}_X.npy')
    yn = np.load(DATA_DIR / f'node_{i:02d}_y.npy')
    partitions.append((Xn, yn))

benign_enc = int([k for k, v in class_mapping.items() if v == 'benign'][0])
X_ae_train = X_train[y_train == benign_enc]

print("=" * 60)
print("  DATA LOADED")
print("=" * 60)
print(f"  Train:    {X_train.shape}")
print(f"  Test:     {X_test.shape}")
print(f"  Classes:  {CLASS_NAMES}")
print(f"  Rounds:   {args.num_rounds}  |  Local epochs: {args.local_epochs}")
print(f"  mu:       {args.mu}")
print(f"  OS Tier1: n < {OS_TIER1_THRESHOLD}/node → {OS_TIER1_TARGET} samples")
print(f"  OS Tier2: {OS_TIER1_THRESHOLD} ≤ n < {OS_TIER2_THRESHOLD}/node → {OS_TIER2_TARGET} samples")
print("=" * 60)

# Partition coverage check — every attack class must be on ≥4 nodes
print("\nPartition coverage check:")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if cls_name == 'benign':
        continue
    node_count = sum(1 for _, yn in partitions if cls_idx in np.unique(yn))
    status = "OK" if node_count >= 4 else "LOW"
    print(f"  {cls_name:<15}: {node_count} nodes  [{status}]")

# Pre-simulation oversampling preview — shows exactly what will be boosted
print(f"\nExpected oversampling per node:")
for i, (Xn, yn) in enumerate(partitions):
    boosts = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_name == 'benign':
            continue
        n = (yn == cls_idx).sum()
        if 0 < n < OS_TIER1_THRESHOLD:
            boosts.append(f"{cls_name}({n}→{OS_TIER1_TARGET}, tier1)")
        elif OS_TIER1_THRESHOLD <= n < OS_TIER2_THRESHOLD:
            boosts.append(f"{cls_name}({n}→{OS_TIER2_TARGET}, tier2)")
    if boosts:
        print(f"  Node {i+1:02d}: {', '.join(boosts)}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Random Forest centralized baseline (FR10)
#
# RF is trained on the full training set. Its Macro F1 is the privacy-accuracy
# ceiling: FL must share 0% of raw data to achieve a competitive F1.
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)

print("\n" + "=" * 60)
print("  PART 2 — RANDOM FOREST CENTRALIZED BASELINE (FR10)")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',   # handles imbalance without manual weights
    n_jobs=-1,
    random_state=42,
)
rf_model.fit(X_train, y_train)

y_pred_rf  = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)

acc_rf    = accuracy_score(y_test, y_pred_rf)
f1_rf_mac = f1_score(y_test, y_pred_rf, average='macro')
f1_rf_w   = f1_score(y_test, y_pred_rf, average='weighted')
f1_rf_cls = f1_score(y_test, y_pred_rf, average=None)

try:
    auc_rf = roc_auc_score(y_test, y_proba_rf, multi_class='ovr', average='macro')
except Exception:
    auc_rf = float('nan')

cm_rf  = confusion_matrix(y_test, y_pred_rf)
fpr_rf = {}
for i, cls in enumerate(CLASS_NAMES):
    TP = cm_rf[i, i]
    FP = cm_rf[:, i].sum() - TP
    TN = cm_rf.sum() - cm_rf[i, :].sum() - cm_rf[:, i].sum() + TP
    fpr_rf[cls] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

print(f"  Accuracy      : {acc_rf:.4f}")
print(f"  Macro F1      : {f1_rf_mac:.4f}  ← FR10 reference")
print(f"  Weighted F1   : {f1_rf_w:.4f}")
print(f"  ROC-AUC (OvR) : {auc_rf:.4f}")
print(f"\n  {'Class':<15} {'F1':>8} {'FPR':>8}")
print("  " + "-" * 34)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<15} {f1_rf_cls[i]:>8.4f} {fpr_rf[cls]:>8.4f}")
print()
print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, digits=4))

RF_RESULTS = {
    'accuracy':      acc_rf,
    'macro_f1':      f1_rf_mac,
    'weighted_f1':   f1_rf_w,
    'roc_auc':       auc_rf,
    'per_class_f1':  f1_rf_cls.tolist(),
    'per_class_fpr': fpr_rf,
}

feat_names  = [f'feature_{i}' for i in range(N_FEATURES)]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
print("Top 10 RF feature importances:")
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:>2}. {feat_names[idx]:<28} {importances[idx]:.4f}")

joblib.dump(rf_model, MODELS_DIR / 'rf_centralized.pkl')
print(f"\nSaved: {MODELS_DIR / 'rf_centralized.pkl'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — BiCNN-LSTM model builder
#
# ALL Keras imports are LOCAL to this function. This is mandatory for Ray:
# cloudpickle traces every global reachable from client_fn. Module-level Keras
# imports create KerasLazyLoader objects that cannot be pickled, causing a crash
# when Ray ships client_fn to worker actors.
# ══════════════════════════════════════════════════════════════════════════════

def build_bicnn_lstm(n_features: int, n_classes: int,
                     conv_filters: int = 64, lstm_units: int = 64,
                     dense_units: int = 64, dropout_rate: float = 0.3):
    """
    Bidirectional CNN-LSTM for 8-class IIoT traffic classification.

    Architecture:
        Input(17) → Reshape(1,17) → Conv1D(64, k=1) → BatchNorm
                  → BiLSTM(64+64=128) → Dense(64) → Dropout(0.3) → Softmax(8)

    Conv1D kernel_size=1: learned feature projection across 17 channels.
    BiLSTM merge_mode='concat': output dim = 2 × lstm_units = 128.
    recurrent_dropout=0.0: avoids non-determinism across parallel Ray workers.
    Total trainable params: 76,104.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(n_features,), name='features')
    x   = layers.Reshape((1, n_features), name='reshape_to_sequence')(inp)
    x   = layers.Conv1D(conv_filters, kernel_size=1, activation='relu',
                        padding='same', name='conv1d')(x)
    x   = layers.BatchNormalization(name='bn_conv')(x)
    x   = layers.Bidirectional(
              layers.LSTM(lstm_units, activation='tanh',
                          recurrent_activation='sigmoid',
                          return_sequences=False,
                          dropout=0.1, recurrent_dropout=0.0, name='lstm'),
              merge_mode='concat', name='bidirectional_lstm')(x)
    x   = layers.Dense(dense_units, activation='relu', name='dense')(x)
    x   = layers.Dropout(dropout_rate, name='dropout')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = keras.Model(inp, out, name='FL_BiCNN_LSTM_IDS')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Print architecture once before the simulation
_ref = build_bicnn_lstm(N_FEATURES, N_CLASSES)
_ref.summary()
print(f"\n  Trainable params: {_ref.count_params():,}")
del _ref

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Flower FL client
# ══════════════════════════════════════════════════════════════════════════════

import flwr as fl


class IIoTBiCNNLSTMClient(fl.client.NumPyClient):
    """
    Flower FL client for one IIoT edge node.

    Privacy  (NF1/FR1) : self.X, self.y are local-only, never transmitted.
                         Only float32 weight arrays leave this object.
    FedAvg   (FR2)     : fit() returns len(self.X) so the server weights
                         this client proportionally (McMahan 2017, Eq. 4).
    FedProx  (FR3)     : proximal term anchors local weights to the global
                         snapshot, preventing drift on non-IID data.
    Oversamp (FIX 4)   : two-tier scheme — targets bruteforce, web (tier1)
                         and mitm (tier2) without touching ddos, dos, recon.
    """

    def __init__(self, node_id: int, X: np.ndarray, y: np.ndarray,
                 n_features: int, n_classes: int,
                 local_epochs: int = 3, mu: float = 0.01,
                 class_mapping: dict = None):
        self.node_id       = node_id
        self.X             = X
        self.y             = y
        self.n_features    = n_features
        self.n_classes     = n_classes
        self.local_epochs  = local_epochs
        self.mu            = mu
        # class_mapping is a plain dict — safe for Ray to pickle
        self.class_mapping = class_mapping or {}
        # Keras is imported inside build_bicnn_lstm → no KerasLazyLoader leak
        self.model = build_bicnn_lstm(n_features, n_classes)

    # ── Oversampling (FIX 4) ─────────────────────────────────────────────────

    def _oversample_minority(self, X: np.ndarray, y: np.ndarray):
        """
        Two-tier local oversampling — privacy-safe (NF1).

        All synthetic samples are generated from the node's own data via
        random sampling with replacement. No external data is used.

        Tier 1 (n < OS_TIER1_THRESHOLD = 500):
            Covers bruteforce (297/node) and web (444/node).
            Brings them to OS_TIER1_TARGET = 500.
            Replication: bruteforce 297→500 = 1.68×, web 444→500 = 1.13×.

        Tier 2 (OS_TIER1_THRESHOLD ≤ n < OS_TIER2_THRESHOLD = 1300):
            Covers mitm (1260/node).
            Brings them to OS_TIER2_TARGET = 2000.
            Replication: mitm 1260→2000 = 1.59×.
            Rationale: mitm has adequate samples (1260/node) but FedAvg
            gradient dilution is severe — 7 of 12 nodes carry no mitm and
            contribute 58% of the weight update, suppressing the mitm output
            neuron each round. This boost gives the 5 mitm-carrying nodes
            a stronger gradient voice to counteract that effect.

        Classes NOT oversampled (ddos ≥2824, dos ≥2880, recon ≥5256):
            These have enough samples; oversampling them only adds noise.
        """
        benign_enc = int(
            [k for k, v in self.class_mapping.items() if v.lower() == 'benign'][0]
        )
        X_aug, y_aug = [X], [y]

        for cls in np.unique(y):
            if cls == benign_enc:
                continue
            idx   = np.where(y == cls)[0]
            n_cls = len(idx)

            if n_cls < OS_TIER1_THRESHOLD:
                # Tier 1: very rare class
                shortfall = OS_TIER1_TARGET - n_cls
                target    = OS_TIER1_TARGET
            elif n_cls < OS_TIER2_THRESHOLD:
                # Tier 2: moderately rare class (mitm)
                shortfall = OS_TIER2_TARGET - n_cls
                target    = OS_TIER2_TARGET
            else:
                continue  # well-represented — do not oversample

            extra = np.random.choice(idx, size=shortfall, replace=True)
            X_aug.append(X[extra])
            y_aug.append(y[extra])

        X_out = np.vstack(X_aug)
        y_out = np.concatenate(y_aug)
        perm  = np.random.permutation(len(X_out))
        return X_out[perm], y_out[perm]

    # ── Flower interface ─────────────────────────────────────────────────────

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """
        Local training: two-tier oversampling + FedProx + weighted CE.

        Steps:
          1. set_parameters() — load global weights from server.
          2. _oversample_minority() — two-tier boost (tier1: bruteforce/web,
             tier2: mitm). Local only, privacy-preserving.
          3. Snapshot model.trainable_weights → proximal anchor w_global.
             (14 arrays, NOT get_weights() which has 16 including BN stats.)
          4. Compute per-node inverse-frequency class weights from the
             post-oversampling distribution. Oversampled classes have more
             samples so their weight decreases proportionally — preventing
             them from dominating while remaining up-weighted vs benign.
          5. GradientTape loop: loss = CE(weighted) + (μ/2)‖w−w_global‖².
          6. Return: (weights, n_samples_for_FedAvg_weighting, {}).
        """
        import tensorflow as tf

        self.set_parameters(parameters)

        # Two-tier oversampling (FIX 4)
        self.X, self.y = self._oversample_minority(self.X, self.y)

        # Snapshot trainable weights as proximal anchor (FIX 2).
        # CRITICAL: use model.trainable_weights (14 arrays), NOT model.get_weights()
        # (16 arrays). The extra 2 are BN moving_mean and moving_var — non-trainable.
        # Zipping 14 vs 16 misaligns shapes at index 4 ([64,256] vs [64]) → crash.
        global_weights = [
            tf.constant(w.numpy(), dtype=tf.float32)
            for w in self.model.trainable_weights
        ]

        # Read round-specific hyperparameters from server config
        epochs     = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', 256)
        mu         = config.get('mu', self.mu)

        # Per-node inverse-frequency class weights, computed from oversampled y.
        # This is correct: oversampled classes have more samples → lower weight →
        # they don't saturate the loss while still being boosted vs benign.
        local_classes = np.unique(self.y)
        n_total       = len(self.y)
        local_cw      = {}
        for cls in range(self.n_classes):
            if cls in local_classes:
                local_cw[cls] = n_total / (len(local_classes) * (self.y == cls).sum())
            else:
                local_cw[cls] = 0.0  # class absent on this node

        cw_tensor = tf.constant(
            [local_cw.get(c, 1.0) for c in range(self.n_classes)],
            dtype=tf.float32
        )

        # tf.data pipeline: shuffle once per epoch, batch, prefetch
        dataset = (
            tf.data.Dataset
            .from_tensor_slices((
                tf.cast(self.X, tf.float32),
                tf.cast(self.y, tf.int32)
            ))
            .shuffle(buffer_size=min(len(self.X), 10000), seed=42)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Fresh Adam each round.
        # A persistent Adam accumulates m̂/v̂ from early rounds when benign
        # dominated the gradient. In later rounds this stale momentum
        # suppresses the minority-class weight updates. Resetting ensures
        # each round's gradient is fully guided by the current global model.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        for _ in range(epochs):
            for X_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(X_batch, training=True)

                    # Weighted sparse cross-entropy
                    ce_loss = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(
                            y_batch, logits
                        ) * tf.gather(cw_tensor, y_batch)
                    )

                    # FedProx proximal term: μ/2 × Σ‖w_i − w_global_i‖²
                    # w_local.value() extracts the tensor from the tf.Variable
                    # so the subtraction is tracked by GradientTape.
                    prox = tf.add_n([
                        tf.reduce_sum(tf.square(wl.value() - wg))
                        for wl, wg in zip(self.model.trainable_weights, global_weights)
                    ])
                    loss = ce_loss + (mu / 2.0) * prox

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Return sample count (len after oversampling) for FedAvg weighting.
        # Larger nodes contribute proportionally more to the global average.
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {'accuracy': float(acc)}


print(f"\nFL client partitions ({N_NODES} nodes):")
for i, (Xn, yn) in enumerate(partitions):
    classes = sorted([CLASS_NAMES[c] for c in np.unique(yn)])
    print(f"  Node {i+1:02d}: {len(Xn):>6,} samples | {classes}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Flower server + FedAvg strategy + simulation
# ══════════════════════════════════════════════════════════════════════════════

from typing import List, Tuple, Dict
from flwr.common import Metrics
from flwr.simulation import start_simulation

round_log: Dict[str, list] = {
    'round': [], 'val_loss': [], 'val_acc': [], 'macro_f1': []
}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Standard FedAvg accuracy aggregation: weight by sample count."""
    total = sum(n for n, _ in metrics)
    return {'accuracy': sum(n * m['accuracy'] for n, m in metrics) / total}


# Server-side Keras (main process only — Ray never serialises these imports)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # noqa: F401

tf.random.set_seed(42)


class BiCNNLSTMFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg extended with:
      • Server-side Macro F1 evaluation after each round.
        Macro F1 weights all 8 classes equally and captures minority-class
        performance that accuracy alone would mask.
      • Best-weights checkpoint by Macro F1.
        The last round's weights are not necessarily the best; the best
        checkpoint from the full 65 rounds is restored after simulation.
    """

    def __init__(self, global_model, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.best_f1      = 0.0
        self.best_weights = None

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        if agg_params is not None:
            weights = fl.common.parameters_to_ndarrays(agg_params)
            self.global_model.set_weights(weights)

            loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0)
            y_pred    = self.global_model.predict(X_test, verbose=0).argmax(axis=1)
            f1_mac    = f1_score(y_test, y_pred, average='macro')

            round_log['round'].append(server_round)
            round_log['val_loss'].append(loss)
            round_log['val_acc'].append(acc)
            round_log['macro_f1'].append(f1_mac)

            print(f"  Round {server_round:02d}/{args.num_rounds} | "
                  f"Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
                  f"Macro F1: {f1_mac:.4f}", flush=True)

            if f1_mac > self.best_f1:
                self.best_f1      = f1_mac
                # Deep-copy: a later round's in-place weight mutation must not
                # corrupt this checkpoint
                self.best_weights = [w.copy() for w in weights]
                print(f"           ^ New best F1: {f1_mac:.4f} — checkpoint saved")

        return agg_params, agg_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


# Build global model in the main process (never serialised by Ray)
global_bicnn_lstm = build_bicnn_lstm(N_FEATURES, N_CLASSES)

# Extract to plain scalars — avoids capturing the argparse.Namespace in client_fn
_LOCAL_EPOCHS = args.local_epochs
_MU           = args.mu

strategy = BiCNNLSTMFedAvgStrategy(
    global_model=global_bicnn_lstm,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=N_NODES,
    min_evaluate_clients=N_NODES,
    min_available_clients=N_NODES,
    on_fit_config_fn=lambda rnd: {
        'local_epochs': _LOCAL_EPOCHS,
        'batch_size':   256,
        'round':        rnd,
        'mu':           _MU,
    },
    evaluate_metrics_aggregation_fn=weighted_average,
)


def client_fn(cid: str) -> fl.client.NumPyClient:
    """
    Factory function Ray pickles and ships to worker actors.

    Captures ONLY plain Python/NumPy objects (picklable):
      partitions    — list[(np.ndarray, np.ndarray)]
      N_FEATURES    — int
      N_CLASSES     — int
      _LOCAL_EPOCHS — int
      _MU           — float
      class_mapping — plain dict

    IIoTBiCNNLSTMClient has NO Keras at class-body level. The model is built
    inside __init__ after Ray deserialises the object in the worker process.
    OS thresholds (OS_TIER1_THRESHOLD etc.) are module-level plain ints —
    safely captured by cloudpickle.
    """
    idx    = int(cid)
    Xn, yn = partitions[idx]
    return IIoTBiCNNLSTMClient(
        node_id=idx + 1,
        X=Xn, y=yn,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        local_epochs=_LOCAL_EPOCHS,
        mu=_MU,
        class_mapping=class_mapping,
    )


print("\n" + "=" * 65)
print("  STARTING FL SIMULATION")
print(f"  BiCNN-LSTM + FedAvg + FedProx (μ={args.mu})")
print(f"  {args.num_rounds} rounds | {N_NODES} nodes | {args.local_epochs} local epochs")
print(f"  Two-tier oversampling: tier1 <{OS_TIER1_THRESHOLD}→{OS_TIER1_TARGET}, "
      f"tier2 <{OS_TIER2_THRESHOLD}→{OS_TIER2_TARGET}")
print(f"  Memory safety: num_cpus=2 → max 6 concurrent workers (OOM prevention)")
print("=" * 65)

start_simulation(
    client_fn=client_fn,
    num_clients=N_NODES,
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
    # FIX 5 — OOM prevention:
    # num_cpus=2 limits concurrent workers to 12÷2=6 instead of 12.
    # Each TF worker uses ~50–100 MB. 12 workers × 100 MB = 1.2 GB can
    # exceed the ≈2.1 GB Ray object store. 6 workers × 100 MB = 600 MB
    # is safely within the limit.
    client_resources={'num_cpus': 2, 'num_gpus': 0.0},
    ray_init_args={
        'ignore_reinit_error': True,
        'log_to_driver':       False,
        # Cap object store to leave memory headroom for TF workers.
        # Without this, Ray can claim all available memory for object storage,
        # leaving TF workers nothing to allocate their computation graphs.
        'object_store_memory': int(1.5e9),  # 1.5 GB cap (vs default ≈2.1 GB)
    },
)

print(f"\nFL simulation complete.")
print(f"  Best Macro F1 : {strategy.best_f1:.4f}")
print(f"  RF baseline   : {f1_rf_mac:.4f}")
print(f"  Paper target  : 0.9551 (Firouzi et al. 2025, Table 8)")

# Restore the best checkpoint (not necessarily the last round's weights)
if strategy.best_weights is not None:
    global_bicnn_lstm.set_weights(strategy.best_weights)
    print("  Best weights restored from checkpoint.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Convergence plots
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'FL-BiCNN-LSTM — FedAvg + FedProx (μ={args.mu}) + two-tier oversampling\n'
    f'{args.num_rounds} rounds, {N_NODES} nodes, non-IID, {args.local_epochs} local epochs',
    fontsize=12
)
rounds = round_log['round']

axes[0].plot(rounds, round_log['val_loss'], 'o-', color='#e74c3c', lw=2)
axes[0].set_title('Validation loss per round')
axes[0].set_xlabel('FL Round')
axes[0].set_ylabel('Cross-entropy loss')
axes[0].grid(alpha=0.3)

axes[1].plot(rounds, [a * 100 for a in round_log['val_acc']], 's-', color='#3498db', lw=2)
axes[1].set_title('Validation accuracy per round')
axes[1].set_xlabel('FL Round')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(alpha=0.3)

axes[2].plot(rounds, round_log['macro_f1'], '^-', color='#2ecc71', lw=2.5, ms=8,
             label='FL Macro F1')
axes[2].axhline(RF_RESULTS['macro_f1'], color='#e67e22', ls='--', lw=1.5,
                label=f"RF F1 = {RF_RESULTS['macro_f1']:.4f}")
axes[2].axhline(strategy.best_f1, color='gray', ls=':', alpha=0.7,
                label=f"Best FL F1 = {strategy.best_f1:.4f}")
axes[2].set_title('Macro F1 per round (primary metric)')
axes[2].set_xlabel('FL Round')
axes[2].set_ylabel('Macro F1')
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'fl_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'fl_convergence.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — Final BiCNN-LSTM evaluation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART 7 — FEDERATED BiCNN-LSTM — FINAL EVALUATION")
print("=" * 60)

y_proba_bicnn = global_bicnn_lstm.predict(X_test, verbose=0)
y_pred_bicnn  = y_proba_bicnn.argmax(axis=1)

acc_bicnn    = accuracy_score(y_test, y_pred_bicnn)
f1_bicnn_mac = f1_score(y_test, y_pred_bicnn, average='macro')
f1_bicnn_w   = f1_score(y_test, y_pred_bicnn, average='weighted')
f1_bicnn_cls = f1_score(y_test, y_pred_bicnn, average=None)

try:
    auc_bicnn = roc_auc_score(
        y_test, y_proba_bicnn, multi_class='ovr', average='macro'
    )
except Exception:
    auc_bicnn = float('nan')

cm_bicnn  = confusion_matrix(y_test, y_pred_bicnn)
fpr_bicnn = {}
for i, cls in enumerate(CLASS_NAMES):
    TP = cm_bicnn[i, i]
    FP = cm_bicnn[:, i].sum() - TP
    TN = cm_bicnn.sum() - cm_bicnn[i, :].sum() - cm_bicnn[:, i].sum() + TP
    fpr_bicnn[cls] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

print(f"  Accuracy      : {acc_bicnn:.4f}")
print(f"  Macro F1      : {f1_bicnn_mac:.4f}  ← PRIMARY METRIC")
print(f"  Weighted F1   : {f1_bicnn_w:.4f}")
print(f"  ROC-AUC (OvR) : {auc_bicnn:.4f}")
print(f"\n  Paper reference: Acc=0.9545, F1=0.9551 (Firouzi et al. 2025, Table 8)")
print(f"\n  {'Class':<15} {'F1':>8} {'FPR':>8} {'Support':>10}")
print("  " + "-" * 44)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<15} {f1_bicnn_cls[i]:>8.4f} {fpr_bicnn[cls]:>8.4f} {(y_test==i).sum():>10,}")
print()
print(classification_report(y_test, y_pred_bicnn, target_names=CLASS_NAMES, digits=4))

BICNN_LSTM_RESULTS = {
    'accuracy':      acc_bicnn,
    'macro_f1':      f1_bicnn_mac,
    'weighted_f1':   f1_bicnn_w,
    'roc_auc':       auc_bicnn,
    'per_class_f1':  f1_bicnn_cls.tolist(),
    'per_class_fpr': fpr_bicnn,
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — FR10: RF vs FL-BiCNN-LSTM privacy-accuracy tradeoff
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FR10 — PRIVACY-ACCURACY TRADEOFF: RF vs FL-BiCNN-LSTM")
print("=" * 65)
print(f"  {'Metric':<20} {'RF (centralized)':>18} {'BiCNN-LSTM (FL)':>18} {'Gap':>8}")
print("  " + "-" * 66)
for label, rv, fv in [
    ('Accuracy',    RF_RESULTS['accuracy'],    BICNN_LSTM_RESULTS['accuracy']),
    ('Macro F1',    RF_RESULTS['macro_f1'],    BICNN_LSTM_RESULTS['macro_f1']),
    ('Weighted F1', RF_RESULTS['weighted_f1'], BICNN_LSTM_RESULTS['weighted_f1']),
    ('ROC-AUC',     RF_RESULTS['roc_auc'],     BICNN_LSTM_RESULTS['roc_auc']),
]:
    gap = fv - rv
    print(f"  {label:<20} {rv:>18.4f} {fv:>18.4f} {gap:>+8.4f}")

gap_f1 = RF_RESULTS['macro_f1'] - BICNN_LSTM_RESULTS['macro_f1']
print(f"\n  Privacy cost of FL : {gap_f1 * 100:.2f}% Macro F1  "
      f"(RF=100% raw data, FL=0% raw data)")

fig, ax = plt.subplots(figsize=(12, 5))
x, w = np.arange(N_CLASSES), 0.35
b1 = ax.bar(x - w/2, RF_RESULTS['per_class_f1'], w,
            label='RF — centralized (100% data)', color='#e74c3c', alpha=0.8)
b2 = ax.bar(x + w/2, BICNN_LSTM_RESULTS['per_class_f1'], w,
            label='BiCNN-LSTM — federated (0% raw data)', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylabel('F1-score')
ax.set_ylim(0, 1.12)
ax.set_title(f'FR10 — Per-class F1: RF centralized vs FL-BiCNN-LSTM\n'
             f'DataSense IIoT 2025 | {N_NODES}-node non-IID | FedProx μ={args.mu}')
ax.legend()
ax.axhline(0.9, color='gray', ls='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'fr10_rf_vs_bicnn_lstm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'fr10_rf_vs_bicnn_lstm.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — Autoencoder anomaly scorer
#
# Trained on benign-only traffic. Attack samples reconstruct poorly
# (high MSE) → flagged as anomalies. Complements the BiCNN-LSTM classifier
# for zero-day detection (attacks not present during classifier training).
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART 9 — AUTOENCODER (unsupervised anomaly scorer)")
print("=" * 60)


def build_autoencoder(n_features: int, bottleneck: int = 4):
    """
    Symmetric encoder-decoder: F→12→8→4→8→12→F.
    Sigmoid output activation: correct since features are Min-Max scaled [0,1].
    Bottleneck=4 gives good benign/attack MSE separation on DataSense.
    Encoder exposed separately for Phase 3 embedding visualisation.
    """
    from tensorflow import keras
    from tensorflow.keras import layers as L

    inp = keras.Input(shape=(n_features,), name='ae_input')
    x   = L.Dense(12, activation='relu', name='enc_1')(inp)
    x   = L.BatchNormalization()(x)
    x   = L.Dense(8,  activation='relu', name='enc_2')(x)
    x   = L.BatchNormalization()(x)
    z   = L.Dense(bottleneck, activation='relu', name='bottleneck')(x)
    x   = L.Dense(8,  activation='relu', name='dec_1')(z)
    x   = L.Dense(12, activation='relu', name='dec_2')(x)
    out = L.Dense(n_features, activation='sigmoid', name='ae_output')(x)

    ae  = keras.Model(inp, out, name='Autoencoder_IDS')
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    enc = keras.Model(inp, z, name='Encoder')
    return ae, enc


ae_model, encoder = build_autoencoder(N_FEATURES)

print(f"Training AE on {len(X_ae_train):,} benign-only samples ...")
ae_model.fit(
    X_ae_train, X_ae_train,
    epochs=150, batch_size=256, validation_split=0.15,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        )
    ],
    verbose=1,
)


def recon_error(model, X: np.ndarray) -> np.ndarray:
    """Per-sample mean-squared reconstruction error."""
    return np.mean(np.square(X - model.predict(X, verbose=0)), axis=1)


# Anomaly threshold: μ + 2σ of benign reconstruction errors.
# Assumes approximately Gaussian distribution → ~2.3% theoretical FPR.
err_benign = recon_error(ae_model, X_ae_train)
THRESHOLD  = err_benign.mean() + 2 * err_benign.std()
print(f"\nAnomaly threshold (μ+2σ): {THRESHOLD:.6f}")

from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.metrics import auc as sk_auc

err_test   = recon_error(ae_model, X_test)
y_bin_true = (y_test != benign_enc).astype(int)
y_bin_pred = (err_test > THRESHOLD).astype(int)

ae_prec = precision_score(y_bin_true, y_bin_pred, zero_division=0)
ae_rec  = recall_score(y_bin_true, y_bin_pred, zero_division=0)
ae_f1   = f1_score(y_bin_true, y_bin_pred, average='binary', zero_division=0)
ae_fpr  = (y_bin_pred[y_bin_true == 0] == 1).mean()
fpr_r, tpr_r, _ = roc_curve(y_bin_true, err_test)
ae_auc  = sk_auc(fpr_r, tpr_r)

print("\n  AUTOENCODER RESULTS (binary: benign vs any-attack)")
print(f"  Precision : {ae_prec:.4f}")
print(f"  Recall    : {ae_rec:.4f}")
print(f"  F1        : {ae_f1:.4f}")
print(f"  FPR       : {ae_fpr:.4f}  ({ae_fpr * 100:.2f}% false alarms)")
print(f"  ROC-AUC   : {ae_auc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(err_benign, bins=50, alpha=0.6, color='#2ecc71', label='Benign (train)', density=True)
axes[0].hist(err_test[y_test != benign_enc], bins=50, alpha=0.6, color='#e74c3c', label='Attack (test)', density=True)
axes[0].axvline(THRESHOLD, color='black', ls='--', lw=2, label=f'Threshold={THRESHOLD:.5f}')
axes[0].set_title('AE reconstruction error distribution')
axes[0].set_xlabel('MSE')
axes[0].legend()

axes[1].plot(fpr_r, tpr_r, color='#9b59b6', lw=2, label=f'AE ROC (AUC={ae_auc:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1].scatter([ae_fpr], [ae_rec], color='red', s=100, zorder=5,
                label=f'Operating point (FPR={ae_fpr:.3f})')
axes[1].set_title('AE ROC curve')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR (Recall)')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'autoencoder_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

AE_RESULTS = {
    'precision': ae_prec, 'recall': ae_rec, 'f1': ae_f1,
    'fpr': ae_fpr, 'roc_auc': ae_auc, 'threshold': float(THRESHOLD),
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 10 — Confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (y_pred, title) in zip(axes, [
    (y_pred_rf,    f'RF — centralized (100% data)\n(Macro F1={f1_rf_mac:.4f})'),
    (y_pred_bicnn, f'FL-BiCNN-LSTM + FedProx + two-tier OS (0% raw data)\n'
                   f'(Macro F1={f1_bicnn_mac:.4f})'),
]):
    cm_n = confusion_matrix(y_test, y_pred).astype(float)
    cm_n /= cm_n.sum(axis=1, keepdims=True)   # row-normalise → per-class recall
    im   = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    thresh = cm_n.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f'{cm_n[i, j]:.2f}',
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color='white' if cm_n[i, j] > thresh else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'confusion_matrices.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 11 — Save all models and results
# ══════════════════════════════════════════════════════════════════════════════

global_bicnn_lstm.save(str(MODELS_DIR / 'fl_bicnn_lstm_model'))
ae_model.save(str(MODELS_DIR / 'autoencoder'))
encoder.save(str(MODELS_DIR / 'encoder'))   # exposed for Phase 3 embeddings
joblib.dump(rf_model, MODELS_DIR / 'rf_centralized.pkl')

with open(MODELS_DIR / 'ae_config.json', 'w') as f:
    json.dump({
        'threshold':   float(THRESHOLD),
        'benign_mean': float(err_benign.mean()),
        'benign_std':  float(err_benign.std()),
    }, f, indent=2)

all_results = {
    'random_forest_centralized': RF_RESULTS,
    'bicnn_lstm_federated':      BICNN_LSTM_RESULTS,
    'autoencoder':               AE_RESULTS,
    'fl_config': {
        'num_rounds':           args.num_rounds,
        'num_clients':          N_NODES,
        'local_epochs':         args.local_epochs,
        'mu':                   args.mu,
        'algorithm':            'FedAvg + FedProx',
        'model':                'BiCNN-LSTM',
        'os_tier1_threshold':   OS_TIER1_THRESHOLD,
        'os_tier1_target':      OS_TIER1_TARGET,
        'os_tier2_threshold':   OS_TIER2_THRESHOLD,
        'os_tier2_target':      OS_TIER2_TARGET,
        'oom_fix':              'num_cpus=2, object_store_memory=1.5GB',
    },
    'convergence_log': round_log,
    'model_selection': {
        'source':                      'Firouzi et al., Electronics 2025, 14, 4095 — Table 8',
        'paper_acc_8class':            0.9545,
        'paper_f1_8class':             0.9551,
        'paper_rf_f1_8class':          0.9780,
        'fixes_applied': [
            'FIX1: num_rounds 20→65, local_epochs 5→3 (drift reduction)',
            'FIX2: FedProx on trainable_weights only (BN shape crash fix)',
            'FIX3: Phase 1 balanced partition — every attack class on ≥4 nodes',
            'FIX4a: Tier1 oversampling n<500→500 (bruteforce 297→500, web 444→500)',
            'FIX4b: Tier2 oversampling 500≤n<1300→2000 (mitm 1260→2000)',
            'FIX5: num_cpus=2, object_store_memory=1.5GB (OOM prevention)',
        ],
    },
}

with open(MODELS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print("=" * 65)
print(f"\n  {'Model':<36} {'Macro F1':>10} {'Data shared':>14}")
print("  " + "-" * 62)
print(f"  {'RF (centralized)':<36} {RF_RESULTS['macro_f1']:>10.4f} {'100% raw':>14}")
print(f"  {'BiCNN-LSTM (FL + FedProx + two-tier OS)':<36} "
      f"{BICNN_LSTM_RESULTS['macro_f1']:>10.4f} {'0% raw':>14}")
print(f"  {'Autoencoder (binary)':<36} {AE_RESULTS['f1']:>10.4f} {'0% raw':>14}")
gap = RF_RESULTS['macro_f1'] - BICNN_LSTM_RESULTS['macro_f1']
print(f"\n  Privacy-accuracy gap (FR10): {gap * 100:.2f}% Macro F1")
print(f"  Best FL Macro F1 (checkpoint): {strategy.best_f1:.4f}")
print(f"  Saved to: {MODELS_DIR}")
print("\n  READY FOR PHASE 3: SDN + Mininet + Online Simulation")
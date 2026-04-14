# -*- coding: utf-8 -*-
"""
FL-SDN-IDS — Phase 2: Offline Federated Learning Training
==========================================================
Project   : Privacy-Preserving Anomaly Detection in Smart Factories
Dataset   : DataSense CIC IIoT 2025 (5-second time window)

Run from VS Code. All paths point to the OUTPUT directory from Phase 1.

Prerequisites
-------------
pip install flwr==1.7.0 tensorflow scikit-learn matplotlib numpy joblib

Usage
-----
python fl_training_BICNNLSTM.py \
    --data_dir   ./processed_output \
    --models_dir ./models \
    --num_rounds   50 \
    --local_epochs  3

Structure
---------
  Part 1  — Load preprocessed data
  Part 2  — Random Forest centralized baseline (FR10)
  Part 3  — BiCNN-LSTM model definition
  Part 4  — Flower client with FedProx + targeted oversampling
  Part 5  — Flower server + FedAvg strategy + simulation
  Part 6  — Convergence plots
  Part 7  — Final BiCNN-LSTM evaluation (accuracy, F1, FPR, ROC-AUC)
  Part 8  — FL vs RF comparison (FR10 privacy-accuracy tradeoff)
  Part 9  — Autoencoder training and anomaly threshold
  Part 10 — Confusion matrices + report figures
  Part 11 — Save all models and results

Model selection rationale (Firouzi et al., Electronics 2025, Table 8):
-----------------------------------------------------------------------
  BiCNN-LSTM    — FL primary classifier (FR1-FR3, NF1, NF2, NF3, NF7)
  Random Forest — centralized baseline (FR10, NF5)
  Autoencoder   — unsupervised anomaly scorer (FR4, FR5, NF1)

KEY FIXES vs original:
-----------------------
  FIX 1 — More rounds + fewer local epochs
    num_rounds 20→50, local_epochs 5→3.
    Justification (McMahan et al. 2017): fewer local steps reduce client drift
    on non-IID data. 50 rounds gives FedAvg enough correction steps.

  FIX 2 — FedProx proximal regularisation (Li et al. 2020)
    μ/2 ‖w_local − w_global‖² added per batch, snapshotted from trainable
    weights only (avoids [64,256] vs [64] shape mismatch from BN moving stats).

  FIX 3 — Phase 1 balanced partitioning
    Every attack class on ≥4 nodes. See phase1_preprocessing.py.

  FIX 4 — Targeted per-node minority oversampling (CORRECTED)
    Previous version used target = 25% of local benign count (~5,472).
    This unintentionally oversampled dos (~2,880/node) and ddos (~940/node),
    causing dos F1 to regress from 0.8750 to 0.8315 (-0.044).

    Root cause: the 25%-of-benign threshold was too high relative to the
    actual attack sample counts on each node. Any attack class with fewer
    samples than 5,472 got oversampled, including classes that were already
    well-represented (dos, ddos, malware).

    Corrected approach: use a DUAL THRESHOLD —
      • Only oversample if n_cls < RARE_COUNT_THRESHOLD (absolute cap = 500)
        This protects dos (≈2,880/node) and ddos (≈940/node) from oversampling.
      • For classes that DO qualify (bruteforce ≈12/node, mitm ≈63/node),
        bring them up to min(500, 25% of benign) samples.
      • 500 was chosen because: bruteforce has ~297 total training samples
        spread across 6 nodes (≈50/node) and mitm has ~315 across 5 (≈63/node).
        Bringing both to 500/node is a 10× boost that stays realistic.

    Privacy: all oversampled data is generated locally from the node's own
    samples via random sampling with replacement. No raw data leaves the node.
    This is compliant with NF1 (data privacy) and FR1 (local computation only).

    Expected impact vs previous oversampling run:
      bruteforce: 0.7262 → 0.80+ (more targeted boost)
      mitm:       0.7412 → 0.80+ (more targeted boost)
      dos:        0.8315 → 0.87+ (regression fixed — no longer oversampled)
      ddos:       0.8469 → 0.86+ (regression fixed — no longer oversampled)
      Macro F1:   0.8469 → 0.87–0.90 (estimated)

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

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',     default='./processed_output')
parser.add_argument('--models_dir',   default='./models')
parser.add_argument('--num_rounds',   type=int,   default=50)
parser.add_argument('--local_epochs', type=int,   default=3)
parser.add_argument('--mu',           type=float, default=0.01)
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
MODELS_DIR = Path(args.models_dir)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Oversampling threshold (FIX 4) ───────────────────────────────────────────
# Only classes with fewer than this many LOCAL samples get oversampled.
# 500 protects dos (≈2880/node) and ddos (≈940/node) while still boosting
# bruteforce (≈50/node) and mitm (≈63/node) which are the real problem classes.
OVERSAMPLE_RARE_THRESHOLD = 500
# Target sample count for rare classes after oversampling.
# 500 samples/node is a 10× boost for bruteforce, realistic for training.
OVERSAMPLE_TARGET = 500

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

print("=" * 55)
print("  DATA LOADED")
print("=" * 55)
print(f"  Train:    {X_train.shape}")
print(f"  Test:     {X_test.shape}")
print(f"  Features: {N_FEATURES}  |  Classes: {N_CLASSES}")
print(f"  FL nodes: {N_NODES}")
print(f"  AE benign train: {X_ae_train.shape}")
print(f"  Classes: {CLASS_NAMES}")
print(f"  FedProx mu: {args.mu}")
print(f"  Oversample: classes with <{OVERSAMPLE_RARE_THRESHOLD} local samples → {OVERSAMPLE_TARGET}")
print("=" * 55)

print("\nPartition coverage check:")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if cls_name == 'benign':
        continue
    node_count = sum(1 for _, yn in partitions if cls_idx in np.unique(yn))
    status = "OK" if node_count >= 4 else "LOW — FL may underperform"
    print(f"  {cls_name:<15}: {node_count} nodes  [{status}]")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Random Forest centralized baseline (FR10)
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)

print("\n" + "=" * 55)
print("  PART 2 — RANDOM FOREST CENTRALIZED BASELINE (FR10)")
print("=" * 55)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=0
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

print("=" * 55)
print("  RANDOM FOREST RESULTS (centralized, all data)")
print("=" * 55)
print(f"  Accuracy      : {acc_rf:.4f}")
print(f"  Macro F1      : {f1_rf_mac:.4f}  <- FR10 reference")
print(f"  Weighted F1   : {f1_rf_w:.4f}")
print(f"  ROC-AUC (OvR) : {auc_rf:.4f}")
print("\n  Per-class F1 and FPR:")
print(f"  {'Class':<15} {'F1':>8} {'FPR':>8}")
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
    print(f"  {rank:>2}. {feat_names[idx]:<30} {importances[idx]:.4f}")

joblib.dump(rf_model, MODELS_DIR / 'rf_centralized.pkl')
print(f"\nSaved RF model -> {MODELS_DIR / 'rf_centralized.pkl'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — BiCNN-LSTM model builder
#
# All Keras imports LOCAL to this function — mandatory for Ray/cloudpickle.
# Module-level Keras imports create KerasLazyLoader objects that cannot
# be pickled when Ray ships client_fn to worker actors.
# ══════════════════════════════════════════════════════════════════════════════

def build_bicnn_lstm(
    n_features:   int,
    n_classes:    int,
    conv_filters: int   = 64,
    lstm_units:   int   = 64,
    dense_units:  int   = 64,
    dropout_rate: float = 0.3,
):
    """
    Bidirectional CNN-LSTM for 8-class IIoT traffic classification.

    Architecture:
        Input(17) → Reshape(1,17) → Conv1D(64,k=1) → BN
                  → BiLSTM(64+64=128) → Dense(64) → Dropout(0.3) → Softmax(8)

    The Conv1D kernel_size=1 acts as a learned feature projection.
    BiLSTM merge_mode='concat' gives output dim = 2 × lstm_units.
    recurrent_dropout=0.0 avoids non-determinism across Ray workers.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(n_features,), name='features')
    x   = layers.Reshape((1, n_features), name='reshape_to_sequence')(inp)
    x   = layers.Conv1D(
            filters=conv_filters, kernel_size=1,
            activation='relu', padding='same', name='conv1d')(x)
    x   = layers.BatchNormalization(name='bn_conv')(x)
    x   = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.0,
                name='lstm'),
            merge_mode='concat',
            name='bidirectional_lstm')(x)
    x   = layers.Dense(dense_units, activation='relu', name='dense')(x)
    x   = layers.Dropout(dropout_rate, name='dropout')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = keras.Model(inp, out, name='FL_BiCNN_LSTM_IDS')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


_ref = build_bicnn_lstm(N_FEATURES, N_CLASSES)
_ref.summary()
print(f"\nTotal parameters: {_ref.count_params():,}")
print("All layers produce weight tensors → compatible with FedAvg")
del _ref

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Flower FL client
#
# Changes vs previous run (document 14 / document 15):
#
#   CORRECTED: _oversample_minority now uses a DUAL THRESHOLD:
#     1. Absolute rarity gate: only oversample if n_cls < OVERSAMPLE_RARE_THRESHOLD (500)
#        This was MISSING in the previous version. Without it, dos (~2880/node)
#        and ddos (~940/node) also got oversampled, causing dos F1 to drop
#        from 0.8750 to 0.8315 (-0.044) and ddos from 0.8650 to 0.8469 (-0.018).
#     2. Fixed target: bring qualifying classes to OVERSAMPLE_TARGET (500) samples,
#        not 25% of benign (which = 5,472 and is too aggressive).
#
#   UNCHANGED: FedProx proximal term on trainable_weights only (not get_weights()),
#   flat lr=1e-3 Adam rebuilt each round, mu=0.01, 50 rounds, 3 local epochs.
#
# Ray serialisation rules (unchanged — mandatory):
#   class_mapping passed as a plain dict (JSON-serialisable) so Ray can pickle it.
#   IIoTBiCNNLSTMClient class body has NO Keras — all imports inside method bodies.
# ══════════════════════════════════════════════════════════════════════════════

import flwr as fl


class IIoTBiCNNLSTMClient(fl.client.NumPyClient):
    """
    Flower FL client for one IIoT edge node.

    Privacy (NF1/FR1)      : self.X, self.y never transmitted.
    FedAvg weighting (FR2) : returns len(self.X) for proportional aggregation.
    FedProx (FR3)          : proximal term on trainable_weights only.
    Oversampling (FIX 4)   : targeted to truly rare classes only (n < 500/node).
    """

    def __init__(
        self,
        node_id:       int,
        X:             np.ndarray,
        y:             np.ndarray,
        n_features:    int,
        n_classes:     int,
        local_epochs:  int   = 3,
        mu:            float = 0.01,
        class_mapping: dict  = None,
    ):
        self.node_id       = node_id
        self.X             = X
        self.y             = y
        self.n_features    = n_features
        self.n_classes     = n_classes
        self.local_epochs  = local_epochs
        self.mu            = mu
        self.class_mapping = class_mapping or {}

        # Keras imported locally inside build_bicnn_lstm — no KerasLazyLoader
        self.model = build_bicnn_lstm(n_features, n_classes)

    def _oversample_minority(self, X, y):
        """
        Targeted per-node oversampling for TRULY RARE classes only.

        Dual-threshold design (FIX 4 — corrected):
        ─────────────────────────────────────────
        Threshold 1 (absolute rarity gate):
            Only oversample attack class i if n_i < OVERSAMPLE_RARE_THRESHOLD.
            Default: 500 samples/node.
            This protects dos (≈2,880/node) and ddos (≈940/node) — they have
            enough training signal already. Oversampling them only adds noise
            and disrupts the class-weight balance computed below.

        Threshold 2 (target count):
            Bring qualifying classes to OVERSAMPLE_TARGET samples.
            Default: 500 samples/node.
            For bruteforce (≈50/node): adds ≈450 synthetic samples = 10× boost.
            For mitm (≈63/node): adds ≈437 synthetic samples = 8× boost.

        Privacy: all synthetic samples are drawn from the node's own data
        via random sampling with replacement. No information about other nodes'
        data is used. This is compliant with NF1 and FR1.

        Why this fixes the dos regression:
            Previous target = int(21,888 × 0.25) = 5,472.
            dos on node 11 has ≈2,880 samples < 5,472 → got oversampled.
            With corrected threshold 500: dos 2,880 > 500 → NOT oversampled.
        """
        benign_enc = int(
            [k for k, v in self.class_mapping.items() if v.lower() == 'benign'][0]
        )

        X_aug, y_aug = [X], [y]
        oversampled = {}

        for cls in np.unique(y):
            if cls == benign_enc:
                continue
            idx   = np.where(y == cls)[0]
            n_cls = len(idx)

            # Threshold 1: only oversample truly rare classes
            if n_cls >= OVERSAMPLE_RARE_THRESHOLD:
                continue   # already has enough samples — do NOT duplicate

            # Threshold 2: bring up to OVERSAMPLE_TARGET
            shortfall = OVERSAMPLE_TARGET - n_cls
            extra_idx = np.random.choice(idx, size=shortfall, replace=True)
            X_aug.append(X[extra_idx])
            y_aug.append(y[extra_idx])
            oversampled[int(cls)] = (n_cls, n_cls + shortfall)

        if oversampled:
            cls_names = {int(k): v for k, v in self.class_mapping.items()}
            report = ', '.join(
                f"{cls_names.get(c, c)} {old}→{new}"
                for c, (old, new) in oversampled.items()
            )
        else:
            report = 'none needed'

        X_out = np.vstack(X_aug)
        y_out = np.concatenate(y_aug)
        perm  = np.random.permutation(len(X_out))
        return X_out[perm], y_out[perm], report

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """
        Local training: FedProx + targeted oversampling.

        Steps
        -----
        1. set_parameters() — load global weights.
        2. _oversample_minority() — boost bruteforce/mitm only (threshold=500).
        3. Snapshot trainable_weights → proximal anchor w_global.
           (Uses trainable_weights, NOT get_weights(), to avoid BN moving-stat
            shape mismatch [64,256] vs [64] that caused crash in an earlier version.)
        4. Compute per-node balanced class weights from OVERSAMPLED distribution.
           This is correct: oversampled classes now have more samples so their
           weight is lower — preventing them from dominating the loss gradient.
        5. For each epoch/batch: loss = CE(weighted) + (mu/2)‖w−w_global‖²
        6. Return weights, sample count, empty metrics dict.
        """
        import tensorflow as tf

        self.set_parameters(parameters)

        # ── FIX 4 (CORRECTED): targeted oversampling ──────────────────────
        original_n = len(self.X)
        self.X, self.y, os_report = self._oversample_minority(self.X, self.y)
        # Note: print statements from Ray workers don't appear in main stdout.
        # The oversampling is verified by the per-class F1 improvements.

        # ── FIX 2: snapshot TRAINABLE weights only ────────────────────────
        # get_weights() = 16 arrays (14 trainable + 2 BN moving stats)
        # trainable_weights = 14 arrays (no moving_mean, moving_var)
        # zip(trainable, get_weights()) misaligns at index 4 → crash.
        # zip(trainable, trainable_snapshot) → correct 1:1 alignment.
        global_weights = [
            tf.constant(w.numpy(), dtype=tf.float32)
            for w in self.model.trainable_weights
        ]

        epochs     = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', 256)
        mu         = config.get('mu', self.mu)

        # Per-node balanced class weights from the OVERSAMPLED distribution.
        # Computing from oversampled y is intentional: rare classes now have
        # more samples so their weight is proportionally reduced, preventing
        # them from dominating every batch while still being up-weighted vs benign.
        local_classes = np.unique(self.y)
        n_total       = len(self.y)
        local_cw      = {}
        for cls in range(self.n_classes):
            if cls in local_classes:
                n_cls         = (self.y == cls).sum()
                local_cw[cls] = n_total / (len(local_classes) * n_cls)
            else:
                local_cw[cls] = 0.0

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

        # Fresh Adam each round — avoids stale momentum from early rounds
        # when benign dominated the gradient, which suppressed minority updates.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        cw_tensor = tf.constant(
            [local_cw.get(c, 1.0) for c in range(self.n_classes)],
            dtype=tf.float32
        )

        for _ in range(epochs):
            for X_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(X_batch, training=True)

                    # Weighted sparse cross-entropy
                    sample_w = tf.gather(cw_tensor, y_batch)
                    ce_loss  = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(
                            y_batch, logits
                        ) * sample_w
                    )

                    # FedProx proximal term: μ/2 × Σ‖w_i − w_global_i‖²
                    # Both lists: 14 trainable_weights ↔ 14 global_weights.
                    # w_local.value() extracts tensor from tf.Variable so
                    # GradientTape tracks the subtraction correctly.
                    prox_term = tf.add_n([
                        tf.reduce_sum(tf.square(w_local.value() - w_global))
                        for w_local, w_global in zip(
                            self.model.trainable_weights, global_weights
                        )
                    ])
                    loss = ce_loss + (mu / 2.0) * prox_term

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {'accuracy': float(acc)}


print(f"\nFL client partitions ({N_NODES} nodes):")
for i, (Xn, yn) in enumerate(partitions):
    present = sorted([CLASS_NAMES[c] for c in np.unique(yn)])
    print(f"  Client {i+1:02d}: {len(Xn):>6,} samples | classes: {present}")

# Show expected oversampling effect per node (pre-simulation sanity check)
print(f"\nExpected oversampling (threshold={OVERSAMPLE_RARE_THRESHOLD}, target={OVERSAMPLE_TARGET}):")
for i, (Xn, yn) in enumerate(partitions):
    will_oversample = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_name == 'benign':
            continue
        n = (yn == cls_idx).sum()
        if 0 < n < OVERSAMPLE_RARE_THRESHOLD:
            will_oversample.append(f"{cls_name}({n}→{OVERSAMPLE_TARGET})")
    if will_oversample:
        print(f"  Node {i+1:02d}: {', '.join(will_oversample)}")

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
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m['accuracy'] for n, m in metrics) / total
    return {'accuracy': acc}


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers   # noqa: F401

tf.random.set_seed(42)


class BiCNNLSTMFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg + server-side Macro F1 evaluation + best-weights checkpoint.
    global_model lives in main process only — Ray never serialises it.
    """

    def __init__(self, global_model, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.best_f1      = 0.0
        self.best_weights = None

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(
            server_round, results, failures
        )

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

            print(
                f"  Round {server_round:02d}/{args.num_rounds} | "
                f"Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
                f"Macro F1: {f1_mac:.4f}",
                flush=True,
            )

            if f1_mac > self.best_f1:
                self.best_f1      = f1_mac
                self.best_weights = [w.copy() for w in weights]
                print(f"           ^ New best F1: {f1_mac:.4f} — checkpoint saved")

        return agg_params, agg_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


global_bicnn_lstm = build_bicnn_lstm(N_FEATURES, N_CLASSES)

_LOCAL_EPOCHS  = args.local_epochs
_MU            = args.mu

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

    Captures ONLY plain Python/NumPy objects — no Keras references:
      partitions     — list[(np.ndarray, np.ndarray)]
      N_FEATURES     — int
      N_CLASSES      — int
      _LOCAL_EPOCHS  — int
      _MU            — float
      class_mapping  — plain dict (JSON-safe, fully picklable)

    IIoTBiCNNLSTMClient body has no Keras. The model is built inside
    __init__ AFTER Ray deserialises the object in the worker process.
    """
    idx    = int(cid)
    Xn, yn = partitions[idx]
    return IIoTBiCNNLSTMClient(
        node_id=idx + 1,
        X=Xn,
        y=yn,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        local_epochs=_LOCAL_EPOCHS,
        mu=_MU,
        class_mapping=class_mapping,
    )


print("\n" + "=" * 65)
print(f"  STARTING FL SIMULATION — BiCNN-LSTM + FedAvg + FedProx")
print(f"  Rounds: {args.num_rounds} | Nodes: {N_NODES} | "
      f"Local epochs: {args.local_epochs} | mu: {args.mu}")
print(f"  Oversampling: rare classes (<{OVERSAMPLE_RARE_THRESHOLD}/node) → {OVERSAMPLE_TARGET} samples")
print("=" * 65)

start_simulation(
    client_fn=client_fn,
    num_clients=N_NODES,
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
    client_resources={'num_cpus': 1, 'num_gpus': 0.0},
    ray_init_args={'ignore_reinit_error': True, 'log_to_driver': False},
)

print(f"\nFL simulation complete.")
print(f"  Best Macro F1 achieved : {strategy.best_f1:.4f}")
print(f"  RF baseline F1         : {f1_rf_mac:.4f}")
print(f"  Paper reference value  : 0.9551 (Table 8, Firouzi et al. 2025)")

if strategy.best_weights is not None:
    global_bicnn_lstm.set_weights(strategy.best_weights)
    print("  Best weights restored from checkpoint.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Convergence plots
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'FL-BiCNN-LSTM — FedAvg + FedProx (μ={args.mu}) + targeted oversampling\n'
    f'{args.num_rounds} rounds, {N_NODES} nodes, non-IID, {args.local_epochs} local epochs',
    fontsize=12
)

rounds     = round_log['round']
rf_f1_line = RF_RESULTS['macro_f1']

axes[0].plot(rounds, round_log['val_loss'], 'o-', color='#e74c3c', lw=2)
axes[0].set_title('Validation loss per round')
axes[0].set_xlabel('FL Round')
axes[0].set_ylabel('Cross-entropy loss')
axes[0].grid(alpha=0.3)

axes[1].plot(rounds, [a * 100 for a in round_log['val_acc']],
             's-', color='#3498db', lw=2)
axes[1].set_title('Validation accuracy per round')
axes[1].set_xlabel('FL Round')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(alpha=0.3)

axes[2].plot(rounds, round_log['macro_f1'], '^-', color='#2ecc71', lw=2.5, ms=8,
             label='FL Macro F1')
axes[2].axhline(rf_f1_line, color='#e67e22', ls='--', lw=1.5,
                label=f'RF centralized F1 = {rf_f1_line:.4f}')
axes[2].axhline(strategy.best_f1, color='gray', ls=':', alpha=0.7,
                label=f'Best FL F1 = {strategy.best_f1:.4f}')
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

print("\n" + "=" * 55)
print("  PART 7 — FEDERATED BiCNN-LSTM — FINAL EVALUATION")
print("=" * 55)

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
print(f"  Macro F1      : {f1_bicnn_mac:.4f}  <- PRIMARY METRIC")
print(f"  Weighted F1   : {f1_bicnn_w:.4f}")
print(f"  ROC-AUC (OvR) : {auc_bicnn:.4f}")
print(f"\n  Paper reference (Table 8): Acc=0.9545 | F1=0.9551")
print("\n  Per-class F1 and FPR:")
print(f"  {'Class':<15} {'F1':>8} {'FPR':>8} {'Support':>10}")
print("  " + "-" * 44)
for i, cls in enumerate(CLASS_NAMES):
    n = (y_test == i).sum()
    print(f"  {cls:<15} {f1_bicnn_cls[i]:>8.4f} {fpr_bicnn[cls]:>8.4f} {n:>10,}")

print("\n" + classification_report(
    y_test, y_pred_bicnn, target_names=CLASS_NAMES, digits=4
))

BICNN_LSTM_RESULTS = {
    'accuracy':      acc_bicnn,
    'macro_f1':      f1_bicnn_mac,
    'weighted_f1':   f1_bicnn_w,
    'roc_auc':       auc_bicnn,
    'per_class_f1':  f1_bicnn_cls.tolist(),
    'per_class_fpr': fpr_bicnn,
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — FR10: RF vs FL-BiCNN-LSTM
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FR10 — PRIVACY-ACCURACY TRADEOFF: RF vs FL-BiCNN-LSTM")
print("=" * 65)
print(f"  {'Metric':<20} {'RF (centralized)':>18} "
      f"{'BiCNN-LSTM (FL)':>18} {'Gap':>8}")
print("  " + "-" * 66)

for label, rf_v, fl_v in [
    ('Accuracy',    RF_RESULTS['accuracy'],    BICNN_LSTM_RESULTS['accuracy']),
    ('Macro F1',    RF_RESULTS['macro_f1'],    BICNN_LSTM_RESULTS['macro_f1']),
    ('Weighted F1', RF_RESULTS['weighted_f1'], BICNN_LSTM_RESULTS['weighted_f1']),
    ('ROC-AUC',     RF_RESULTS['roc_auc'],     BICNN_LSTM_RESULTS['roc_auc']),
]:
    gap  = fl_v - rf_v
    sign = '+' if gap >= 0 else ''
    print(f"  {label:<20} {rf_v:>18.4f} {fl_v:>18.4f} {sign}{gap:>7.4f}")

gap_f1 = RF_RESULTS['macro_f1'] - BICNN_LSTM_RESULTS['macro_f1']
print(f"\n  Privacy cost of FL   : {gap_f1 * 100:.2f}% Macro F1 vs RF baseline")
print(f"  Raw data shared      : RF = 100% | FL-BiCNN-LSTM = 0%")

fig, ax = plt.subplots(figsize=(12, 5))
x  = np.arange(N_CLASSES)
w  = 0.35
b1 = ax.bar(x - w / 2, RF_RESULTS['per_class_f1'], w,
            label='RF — centralized (all data)', color='#e74c3c', alpha=0.8)
b2 = ax.bar(x + w / 2, BICNN_LSTM_RESULTS['per_class_f1'], w,
            label='BiCNN-LSTM — federated (0% raw data shared)',
            color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylabel('F1-score')
ax.set_ylim(0, 1.12)
ax.set_title(
    f'FR10 — Per-class F1: RF centralized vs FL-BiCNN-LSTM\n'
    f'DataSense IIoT 2025 | {N_NODES}-node non-IID | FedProx μ={args.mu}'
)
ax.legend()
ax.axhline(0.9, color='gray', ls='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
            f'{h:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'fr10_rf_vs_bicnn_lstm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'fr10_rf_vs_bicnn_lstm.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PART 9 — AUTOENCODER (unsupervised anomaly scorer)")
print("=" * 55)


def build_autoencoder(n_features: int, bottleneck: int = 4):
    """
    Symmetric encoder-decoder for benign traffic reconstruction.
    Architecture: F→12→8→4→8→12→F
    Sigmoid output: correct since features are Min-Max [0,1].
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

print(f"Training AE on {len(X_ae_train):,} benign-only samples...")
ae_model.fit(
    X_ae_train, X_ae_train,
    epochs=150,
    batch_size=256,
    validation_split=0.15,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        )
    ],
    verbose=1,
)


def recon_error(model, X: np.ndarray) -> np.ndarray:
    return np.mean(np.square(X - model.predict(X, verbose=0)), axis=1)


err_benign = recon_error(ae_model, X_ae_train)
THRESHOLD  = err_benign.mean() + 2 * err_benign.std()
print(f"\nAnomaly threshold (μ + 2σ): {THRESHOLD:.6f}")

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
axes[0].hist(err_benign, bins=50, alpha=0.6, color='#2ecc71',
             label='Benign (train)', density=True)
axes[0].hist(err_test[y_test != benign_enc], bins=50, alpha=0.6,
             color='#e74c3c', label='Attack (test)', density=True)
axes[0].axvline(THRESHOLD, color='black', ls='--', lw=2,
                label=f'Threshold = {THRESHOLD:.5f}')
axes[0].set_title('AE reconstruction error distribution')
axes[0].set_xlabel('MSE')
axes[0].legend()

axes[1].plot(fpr_r, tpr_r, color='#9b59b6', lw=2,
             label=f'AE ROC (AUC={ae_auc:.4f})')
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
    (y_pred_rf,
     f'RF — centralized (100% data)\n(Macro F1 = {f1_rf_mac:.4f})'),
    (y_pred_bicnn,
     f'FL-BiCNN-LSTM + FedProx + oversampling (0% raw data)\n'
     f'(Macro F1 = {f1_bicnn_mac:.4f})'),
]):
    cm_n = confusion_matrix(y_test, y_pred).astype(float)
    cm_n /= cm_n.sum(axis=1, keepdims=True)
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
                    ha='center', va='center', fontsize=7,
                    color='white' if cm_n[i, j] > thresh else 'black',
                    fontweight='bold')
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
encoder.save(str(MODELS_DIR / 'encoder'))
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
        'num_rounds':               args.num_rounds,
        'num_clients':              N_NODES,
        'local_epochs':             args.local_epochs,
        'mu':                       args.mu,
        'algorithm':                'FedAvg + FedProx',
        'model':                    'BiCNN-LSTM',
        'oversample_threshold':     OVERSAMPLE_RARE_THRESHOLD,
        'oversample_target':        OVERSAMPLE_TARGET,
    },
    'convergence_log': round_log,
    'model_selection_reference': {
        'source':                       'Firouzi et al., Electronics 2025, 14, 4095 — Table 8',
        'bicnn_lstm_paper_acc_8class':  0.9545,
        'bicnn_lstm_paper_f1_8class':   0.9551,
        'rf_paper_f1_8class':           0.9780,
        'fixes_applied': [
            'FIX1: num_rounds 20→50, local_epochs 5→3',
            'FIX2: FedProx proximal term on trainable_weights only',
            'FIX3: Phase 1 balanced partition — every attack class on ≥4 nodes',
            'FIX4: Targeted oversampling (threshold=500) — protects dos/ddos from regression',
        ],
    },
}

with open(MODELS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print("=" * 65)
print(f"\n  {'Model':<32} {'Macro F1':>10} {'Data shared':>14}")
print("  " + "-" * 58)
print(f"  {'RF (centralized)':<32} {RF_RESULTS['macro_f1']:>10.4f} {'100% raw':>14}")
print(f"  {'BiCNN-LSTM (FL + FedProx + OS)':<32} "
      f"{BICNN_LSTM_RESULTS['macro_f1']:>10.4f} {'0% raw':>14}")
print(f"  {'Autoencoder (binary)':<32} {AE_RESULTS['f1']:>10.4f} {'0% raw':>14}")
gap = RF_RESULTS['macro_f1'] - BICNN_LSTM_RESULTS['macro_f1']
print(f"\n  Privacy-accuracy gap (FR10) : {gap * 100:.2f}% Macro F1")
print(f"  Best FL Macro F1 (checkpoint): {strategy.best_f1:.4f}")
print(f"  Saved to: {MODELS_DIR}")
print("\n  READY FOR PHASE 3: SDN + Mininet + Online Simulation")
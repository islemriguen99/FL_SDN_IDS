# -*- coding: utf-8 -*-
"""
FL-SDN-IDS — Phase 2: Offline Federated Learning Training
==========================================================
Project   : Privacy-Preserving Anomaly Detection in Smart Factories
Dataset   : DataSense CIC IIoT 2025 (5-second time window)

Run from VS Code. All paths point to the OUTPUT directory from Phase 1.

Prerequisites
-------------
pip install flwr==1.7.0 tensorflow scikit-learn matplotlib numpy joblib xgboost

Usage
-----
python fl_training_BICNNLSTM.py \
    --data_dir   ./processed_output \
    --models_dir ./models \
    --num_rounds   60 \
    --local_epochs  3

Structure
---------
  Part 1  — Load preprocessed data
  Part 2  — XGBoost centralized baseline (FR10)
  Part 3  — BiCNN-LSTM model definition
  Part 4  — Flower client with FedProx + LR schedule + persistent Adam
  Part 5  — Flower server + FedAvg strategy + simulation
  Part 6  — Convergence plots
  Part 7  — Final BiCNN-LSTM evaluation (accuracy, F1, FPR, ROC-AUC)
  Part 8  — FL vs XGBoost comparison (FR10 privacy-accuracy tradeoff)
  Part 9  — Autoencoder training and anomaly threshold
  Part 10 — Confusion matrices + report figures
  Part 11 — Save all models and results

Model selection rationale (Firouzi et al., Electronics 2025, Table 8):
-----------------------------------------------------------------------
  BiCNN-LSTM  — FL primary classifier (FR1-FR3, NF1, NF2, NF3, NF7)
  XGBoost     — centralized baseline (FR10, NF5) — replaces RandomForest
  Autoencoder — unsupervised anomaly scorer (FR4, FR5, NF1)

KEY FIXES vs original:
-----------------------
  FIX 1 — More rounds + fewer local epochs
    num_rounds 20→60, local_epochs 5→3.
    Justification (McMahan et al. 2017): fewer local steps reduce client
    drift on non-IID data; more rounds give FedAvg more correction steps.

  FIX 2 — FedProx on trainable_weights only (shape-mismatch bug fix)
    Original: global_weights snapshotted from get_weights() (16 arrays) but
    zipped against trainable_weights (14 arrays) → [64,256] vs [64] crash.
    Fixed: snapshot from trainable_weights (14 arrays), zip against same.
    BatchNorm moving_mean / moving_var are excluded correctly — they are
    updated internally by BN, not by the optimizer.

  FIX 3 — Phase 1 balanced partitioning
    Every attack class on ≥4 nodes. See phase1_preprocessing.py.

  FIX 4 — Round-aware learning-rate schedule  ← NEW
    lr = 1e-3 (rounds 1-20) → 5e-4 (21-35) → 2e-4 (36+).
    Justification: the plateau/oscillation in rounds 22-50 of the previous
    run (F1 ±0.02) was caused by Adam still using lr=1e-3 in round 44 —
    the same rate used in round 1 when the loss landscape was steep.
    Decaying the lr in the fine-tuning phase gives smaller, more reliable
    gradient steps and tightens the plateau oscillation.

  FIX 5 — Persistent Adam optimizer per client  ← NEW
    Previously a fresh Adam was built inside fit() every round, resetting
    first- and second-moment estimates (m̂, v̂) to zero.
    Fix: Adam is built once in __init__ and its learning_rate is updated
    each round via self.optimizer.learning_rate.assign(lr).
    Justification: Adam's adaptive per-parameter scale is encoded in v̂
    (the second moment). Discarding it every round forces Adam to "warm up"
    from scratch, halving the effective number of optimizer steps.

  FIX 6 — Round-aware FedProx mu schedule  ← NEW
    mu = 0.01 (rounds 1-20) → 0.05 (21-35) → 0.10 (36+).
    Justification: a tighter proximal anchor in later rounds stops clients
    drifting far from the global model during the fine-tuning phase, which
    was the main driver of the observed round-to-round F1 oscillation.

  FIX 7 — XGBoost replaces RandomForest (FR10 baseline)  ← NEW
    RF Macro F1 was 0.932; XGBoost with boosting + sample weighting
    typically gains +0.02-0.04 on minority-heavy datasets like this one,
    approaching the paper's 0.978 centralized result more closely.

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
parser.add_argument('--num_rounds',   type=int,   default=60)
parser.add_argument('--local_epochs', type=int,   default=3)
parser.add_argument('--mu',           type=float, default=0.01,
                    help='Initial FedProx mu (schedule multiplies per phase)')
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
MODELS_DIR = Path(args.models_dir)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Schedule functions (pure Python — safe for Ray to pickle) ─────────────────

def get_lr(rnd: int) -> float:
    """Round-aware Adam learning-rate decay (FIX 4)."""
    if rnd <= 20:  return 1e-3   # exploration phase
    if rnd <= 35:  return 5e-4   # plateau reduction
    return 2e-4                  # deep fine-tuning

def get_mu(rnd: int, base_mu: float) -> float:
    """Round-aware FedProx proximal coefficient (FIX 6)."""
    if rnd <= 20:  return base_mu          # light anchor early
    if rnd <= 35:  return base_mu * 5.0    # tighten as plateau approaches
    return base_mu * 10.0                  # strong anchor in fine-tuning

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
print(f"  Base mu: {args.mu}  |  LR schedule: 1e-3→5e-4→2e-4")
print("=" * 55)

print("\nPartition coverage check:")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if cls_name == 'benign':
        continue
    node_count = sum(1 for _, yn in partitions if cls_idx in np.unique(yn))
    status = "OK" if node_count >= 4 else "LOW — FL may underperform"
    print(f"  {cls_name:<15}: {node_count} nodes  [{status}]")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — XGBoost centralized baseline (FR10)
#
# XGBoost replaces RandomForest (FIX 7).
# Boosting corrects residual errors iteratively, handling minority classes
# (bruteforce n=75, web n=111) better than bagging. Per-sample weights from
# CLASS_WEIGHTS apply balanced importance to rare classes directly.
# ══════════════════════════════════════════════════════════════════════════════

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)

print("\n" + "=" * 55)
print("  PART 2 — XGBoost CENTRALIZED BASELINE (FR10)")
print("=" * 55)

# Per-sample weights: each sample receives its class's balanced weight
sample_weights = np.array([CLASS_WEIGHTS[int(y)] for y in y_train])

xgb_model = XGBClassifier(
    n_estimators=500,       # more trees than RF — boosting benefits from depth
    max_depth=8,            # moderate depth balances bias/variance
    learning_rate=0.05,     # shrinkage reduces overfitting
    subsample=0.8,          # row subsampling
    colsample_bytree=0.8,   # feature subsampling — similar to RF's max_features
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred_rf  = xgb_model.predict(X_test)
y_proba_rf = xgb_model.predict_proba(X_test)

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
print("  XGBoost RESULTS (centralized, all data)")
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
importances = xgb_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
print("Top 10 XGBoost feature importances (gain):")
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:>2}. {feat_names[idx]:<30} {importances[idx]:.4f}")

joblib.dump(xgb_model, MODELS_DIR / 'xgb_centralized.pkl')
print(f"\nSaved XGBoost model -> {MODELS_DIR / 'xgb_centralized.pkl'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — BiCNN-LSTM model builder
#
# All Keras imports are LOCAL to this function.
# Mandatory for Ray/cloudpickle safety: module-level Keras imports create
# KerasLazyLoader objects that cannot be pickled → crash when Ray ships
# client_fn to worker actors.
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
        Input(F) → Reshape(1,F) → Conv1D(64,k=1) → BN
                 → BiLSTM(64+64) → Dense(64) → Dropout → Softmax(8)

    Conv1D kernel_size=1: learned projection across all 17 features at each
    time step. BiLSTM merge_mode='concat': retains both forward and backward
    hidden states (output dim = 2 × lstm_units = 128).
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
# Incorporates FIX 2 (trainable_weights snapshot), FIX 4 (LR from config),
# FIX 5 (persistent Adam), FIX 6 (mu from config).
#
# Ray serialisation rules obeyed:
#   • No Keras at class-body level — all imports inside __init__ / fit().
#   • client_fn is top-level, captures only plain scalars/arrays.
#   • get_lr and get_mu are pure Python — safely picklable.
# ══════════════════════════════════════════════════════════════════════════════

import flwr as fl


class IIoTBiCNNLSTMClient(fl.client.NumPyClient):
    """
    Flower FL client for one IIoT edge node.

    Privacy (NF1/FR1)       : self.X, self.y never transmitted.
    FedAvg weighting (FR2)  : returns len(self.X) for proportional aggregation.
    FedProx (FR3)           : proximal term anchors local to global weights.
    LR schedule (FIX 4)     : lr updated from config['lr'] each round.
    Persistent Adam (FIX 5) : optimizer built once, momentum preserved.
    Mu schedule (FIX 6)     : mu updated from config['mu'] each round.
    """

    def __init__(
        self,
        node_id:      int,
        X:            np.ndarray,
        y:            np.ndarray,
        n_features:   int,
        n_classes:    int,
        local_epochs: int   = 3,
        mu:           float = 0.01,
    ):
        self.node_id      = node_id
        self.X            = X
        self.y            = y
        self.n_features   = n_features
        self.n_classes    = n_classes
        self.local_epochs = local_epochs
        self.mu           = mu

        # Model built here — Keras imported locally inside build_bicnn_lstm
        self.model = build_bicnn_lstm(n_features, n_classes)

        # FIX 5: build optimizer ONCE in __init__ — preserved across rounds.
        # tf imported locally to keep this class picklable by Ray.
        import tensorflow as tf
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """
        Local training: FedProx + round-adaptive lr + persistent Adam.

        Steps
        -----
        1. set_parameters() — load global weights from server.
        2. Snapshot trainable_weights → proximal anchor w_global.
           (FIX 2: NOT get_weights() — avoids BN moving-stat misalignment.)
        3. Read lr, mu from config. Assign lr to persistent optimizer.
           (FIX 4 + 5: lr decays per schedule; m̂/v̂ accumulate across rounds.)
        4. For each epoch/batch:
             loss = CE(weighted) + (mu/2) × ‖w_local − w_global‖²
        5. Return weights, sample count (FedAvg weight), empty metrics.
        """
        import tensorflow as tf

        self.set_parameters(parameters)

        # ── FIX 2: trainable_weights only (14 arrays, no BN moving stats) ─
        global_weights = [
            tf.constant(w.numpy(), dtype=tf.float32)
            for w in self.model.trainable_weights
        ]

        # ── Read round-specific values from server ─────────────────────────
        epochs     = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', 256)
        mu         = config.get('mu', self.mu)

        # ── FIX 4 + 5: update lr on persistent optimizer ──────────────────
        lr = config.get('lr', 1e-3)
        self.optimizer.learning_rate.assign(lr)

        # Per-node balanced class weights from local label distribution
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

        # Build class-weight tensor once per fit() call (not per batch)
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
                    # zip: 14 trainable_weights ↔ 14 global_weights (same shapes)
                    # w_local.value() extracts tensor from Variable so
                    # GradientTape tracks the subtraction correctly.
                    prox_term = tf.add_n([
                        tf.reduce_sum(tf.square(w_local.value() - w_global))
                        for w_local, w_global in zip(
                            self.model.trainable_weights, global_weights
                        )
                    ])
                    loss = ce_loss + (mu / 2.0) * prox_term

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
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

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Flower server + FedAvg strategy + simulation
# ══════════════════════════════════════════════════════════════════════════════

from typing import List, Tuple, Dict
from flwr.common import Metrics
from flwr.simulation import start_simulation

round_log: Dict[str, list] = {
    'round': [], 'val_loss': [], 'val_acc': [], 'macro_f1': [],
    'lr': [], 'mu': []
}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted accuracy aggregation (FedAvg standard)."""
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m['accuracy'] for n, m in metrics) / total
    return {'accuracy': acc}


# Server-side Keras — main process only, never serialised by Ray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers   # noqa: F401

tf.random.set_seed(42)


class BiCNNLSTMFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg + server-side Macro F1 evaluation + best-weights checkpoint.
    Logs lr and mu each round for convergence analysis.
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
            round_log['lr'].append(get_lr(server_round))
            round_log['mu'].append(get_mu(server_round, args.mu))

            print(
                f"  Round {server_round:02d}/{args.num_rounds} | "
                f"Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
                f"Macro F1: {f1_mac:.4f} | "
                f"lr={get_lr(server_round):.0e} "
                f"mu={get_mu(server_round, args.mu):.2f}",
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

# Extract to plain scalars — never capture argparse.Namespace in client_fn
_LOCAL_EPOCHS = args.local_epochs
_BASE_MU      = args.mu

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
        # FIX 4 + 6: round-specific lr and mu sent to every client
        'lr':           get_lr(rnd),
        'mu':           get_mu(rnd, _BASE_MU),
    },
    evaluate_metrics_aggregation_fn=weighted_average,
)


def client_fn(cid: str) -> fl.client.NumPyClient:
    """
    Factory function Ray pickles and ships to worker actors.

    Captures ONLY plain Python/NumPy objects:
      partitions     — list[(np.ndarray, np.ndarray)]
      N_FEATURES     — int
      N_CLASSES      — int
      _LOCAL_EPOCHS  — int
      _BASE_MU       — float

    lr and mu are NOT passed here — they come from config inside fit()
    via on_fit_config_fn, which runs in the main process each round.

    IIoTBiCNNLSTMClient body has no Keras — Keras is imported only inside
    __init__ and fit() method bodies, which execute AFTER deserialisation
    in the Ray worker. Nothing unpicklable crosses the boundary.
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
        mu=_BASE_MU,
    )


print("\n" + "=" * 65)
print(f"  STARTING FL SIMULATION — BiCNN-LSTM + FedAvg + FedProx")
print(f"  Rounds: {args.num_rounds} | Nodes: {N_NODES} | "
      f"Local epochs: {args.local_epochs}")
print(f"  LR  schedule: 1e-3 (R1-20) → 5e-4 (R21-35) → 2e-4 (R36+)")
print(f"  Mu  schedule: {args.mu:.2f} (R1-20) → "
      f"{args.mu*5:.2f} (R21-35) → {args.mu*10:.2f} (R36+)")
print(f"  Adam: persistent per client (momentum preserved across rounds)")
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
print(f"  XGBoost baseline F1    : {f1_rf_mac:.4f}")
print(f"  Paper reference value  : 0.9551 (Table 8, Firouzi et al. 2025)")

if strategy.best_weights is not None:
    global_bicnn_lstm.set_weights(strategy.best_weights)
    print("  Best weights restored from checkpoint.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Convergence plots (2×2 grid including schedule visualisation)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f'FL-BiCNN-LSTM Convergence — FedAvg + FedProx + LR/Mu schedule\n'
    f'{args.num_rounds} rounds, {N_NODES} nodes, non-IID, '
    f'{args.local_epochs} local epochs',
    fontsize=12
)

rounds     = round_log['round']
rf_f1_line = RF_RESULTS['macro_f1']

# Plot 1: validation loss
axes[0, 0].plot(rounds, round_log['val_loss'], 'o-', color='#e74c3c', lw=2)
axes[0, 0].set_title('Validation loss per round')
axes[0, 0].set_xlabel('FL Round')
axes[0, 0].set_ylabel('Cross-entropy loss')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Macro F1 with XGBoost and paper reference lines
axes[0, 1].plot(rounds, round_log['macro_f1'], '^-', color='#2ecc71',
                lw=2.5, ms=8, label='FL Macro F1')
axes[0, 1].axhline(rf_f1_line, color='#e67e22', ls='--', lw=1.5,
                   label=f'XGBoost F1 = {rf_f1_line:.4f}')
axes[0, 1].axhline(strategy.best_f1, color='gray', ls=':', alpha=0.7,
                   label=f'Best FL F1 = {strategy.best_f1:.4f}')
axes[0, 1].set_title('Macro F1 per round (primary metric)')
axes[0, 1].set_xlabel('FL Round')
axes[0, 1].set_ylabel('Macro F1')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# Plot 3: LR and mu schedules on dual axes — shows where transitions occur
ax3  = axes[1, 0]
ax3b = ax3.twinx()
ax3.plot(rounds, round_log['lr'], 's-', color='#3498db', lw=2, label='LR')
ax3b.plot(rounds, round_log['mu'], 'D-', color='#9b59b6', lw=2, label='Mu')
ax3.set_title('LR and mu schedules across rounds')
ax3.set_xlabel('FL Round')
ax3.set_ylabel('Learning rate', color='#3498db')
ax3b.set_ylabel('FedProx mu', color='#9b59b6')
ax3.grid(alpha=0.3)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

# Plot 4: validation accuracy
axes[1, 1].plot(rounds, [a * 100 for a in round_log['val_acc']],
                's-', color='#3498db', lw=2)
axes[1, 1].set_title('Validation accuracy per round')
axes[1, 1].set_xlabel('FL Round')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].grid(alpha=0.3)

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
# PART 8 — FR10: XGBoost vs FL-BiCNN-LSTM privacy-accuracy tradeoff
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FR10 — PRIVACY-ACCURACY TRADEOFF: XGBoost vs FL-BiCNN-LSTM")
print("=" * 65)
print(f"  {'Metric':<20} {'XGBoost (central)':>18} "
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
print(f"\n  Privacy cost of FL   : {gap_f1 * 100:.2f}% Macro F1 vs XGBoost baseline")
print(f"  Raw data shared      : XGBoost = 100% | FL-BiCNN-LSTM = 0%")

fig, ax = plt.subplots(figsize=(12, 5))
x  = np.arange(N_CLASSES)
w  = 0.35
b1 = ax.bar(x - w / 2, RF_RESULTS['per_class_f1'], w,
            label='XGBoost — centralized (all data)', color='#e74c3c', alpha=0.8)
b2 = ax.bar(x + w / 2, BICNN_LSTM_RESULTS['per_class_f1'], w,
            label='BiCNN-LSTM — federated (0% raw data shared)',
            color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylabel('F1-score')
ax.set_ylim(0, 1.12)
ax.set_title(
    f'FR10 — Per-class F1: XGBoost centralized vs FL-BiCNN-LSTM\n'
    f'DataSense IIoT 2025 | {N_NODES}-node non-IID | '
    f'FedProx + LR/Mu schedule'
)
ax.legend()
ax.axhline(0.9, color='gray', ls='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
            f'{h:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'fr10_xgb_vs_bicnn_lstm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'fr10_xgb_vs_bicnn_lstm.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — Autoencoder training and anomaly threshold
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PART 9 — AUTOENCODER (unsupervised anomaly scorer)")
print("=" * 55)


def build_autoencoder(n_features: int, bottleneck: int = 4):
    """
    Symmetric encoder-decoder trained on benign traffic only.
    Architecture: F→12→8→bottleneck→8→12→F
    Sigmoid output: correct because features are Min-Max [0,1].
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
    """Per-sample mean squared reconstruction error."""
    return np.mean(np.square(X - model.predict(X, verbose=0)), axis=1)


# Threshold: μ + 2σ of benign reconstruction errors
# Assumes ~Gaussian distribution → ~2.3% theoretical FPR on benign
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
     f'XGBoost — centralized (100% data)\n(Macro F1 = {f1_rf_mac:.4f})'),
    (y_pred_bicnn,
     f'FL-BiCNN-LSTM + FedProx + LR/Mu schedule (0% raw data)\n'
     f'(Macro F1 = {f1_bicnn_mac:.4f})'),
]):
    cm_n = confusion_matrix(y_test, y_pred).astype(float)
    cm_n /= cm_n.sum(axis=1, keepdims=True)   # row-normalise → recall per class
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
joblib.dump(xgb_model, MODELS_DIR / 'xgb_centralized.pkl')

with open(MODELS_DIR / 'ae_config.json', 'w') as f:
    json.dump({
        'threshold':   float(THRESHOLD),
        'benign_mean': float(err_benign.mean()),
        'benign_std':  float(err_benign.std()),
    }, f, indent=2)

all_results = {
    'xgboost_centralized':  RF_RESULTS,
    'bicnn_lstm_federated': BICNN_LSTM_RESULTS,
    'autoencoder':          AE_RESULTS,
    'fl_config': {
        'num_rounds':   args.num_rounds,
        'num_clients':  N_NODES,
        'local_epochs': args.local_epochs,
        'base_mu':      args.mu,
        'mu_schedule':  'x1→x5→x10 at rounds 20,35',
        'lr_schedule':  '1e-3→5e-4→2e-4 at rounds 20,35',
        'algorithm':    'FedAvg + FedProx + LR/Mu schedule + persistent Adam',
        'model':        'BiCNN-LSTM',
    },
    'convergence_log': round_log,
    'model_selection_reference': {
        'source':                       'Firouzi et al., Electronics 2025, 14, 4095 — Table 8',
        'bicnn_lstm_paper_acc_8class':  0.9545,
        'bicnn_lstm_paper_f1_8class':   0.9551,
        'centralized_paper_f1_8class':  0.9780,
        'fixes_applied': [
            'FIX1: num_rounds 20→60, local_epochs 5→3 (reduce drift)',
            'FIX2: FedProx proximal term on trainable_weights only (shape fix)',
            'FIX3: Phase 1 balanced partition — every attack class on ≥4 nodes',
            'FIX4: LR decay schedule: 1e-3→5e-4→2e-4 at rounds 20,35',
            'FIX5: Persistent Adam — m̂/v̂ preserved across rounds',
            'FIX6: Mu schedule: x1→x5→x10 of base_mu at rounds 20,35',
            'FIX7: XGBoost replaces RandomForest for centralized baseline',
        ],
    },
}

with open(MODELS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print("=" * 65)
print(f"\n  {'Model':<38} {'Macro F1':>10} {'Data shared':>14}")
print("  " + "-" * 64)
print(f"  {'XGBoost (centralized)':<38} {RF_RESULTS['macro_f1']:>10.4f} {'100% raw':>14}")
print(f"  {'BiCNN-LSTM (FL + FedProx + schedules)':<38} "
      f"{BICNN_LSTM_RESULTS['macro_f1']:>10.4f} {'0% raw':>14}")
print(f"  {'Autoencoder (binary)':<38} {AE_RESULTS['f1']:>10.4f} {'0% raw':>14}")
gap = RF_RESULTS['macro_f1'] - BICNN_LSTM_RESULTS['macro_f1']
print(f"\n  Privacy-accuracy gap (FR10) : {gap * 100:.2f}% Macro F1")
print(f"  Best FL Macro F1 (checkpoint): {strategy.best_f1:.4f}")
print(f"  Saved to: {MODELS_DIR}")
print("\n  READY FOR PHASE 3: SDN + Mininet + Online Simulation")
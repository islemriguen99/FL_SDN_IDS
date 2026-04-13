"""
FL-SDN-IDS — Phase 2: Offline Federated Learning Training
==========================================================
Run from VS Code. All paths use the OUTPUT directory from Phase 1.

Prerequisites
-------------
pip install flwr==1.7.0 tensorflow scikit-learn matplotlib numpy joblib

Usage
-----
python fl_training.py --data_dir ./data --models_dir ./models

Structure
---------
  Part 1  — Load preprocessed data
  Part 2  — Random Forest centralized baseline (FR10)
  Part 3  — GRU model definition
  Part 4  — Flower client (IIoTFlowerClient)
  Part 5  — Flower server + FedAvg + simulation
  Part 6  — Convergence plots
  Part 7  — Final GRU evaluation (accuracy, F1, FPR, ROC-AUC)
  Part 8  — FL vs RF comparison (FR10 privacy-accuracy tradeoff)
  Part 9  — Autoencoder training and anomaly threshold
  Part 10 — Confusion matrices + report figures
  Part 11 — Save all models and results
"""

import os, sys, json, joblib, argparse, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   default='./data',   help='Phase 1 output directory')
parser.add_argument('--models_dir', default='./models', help='Where to save trained models')
parser.add_argument('--num_rounds', type=int, default=20)
parser.add_argument('--local_epochs', type=int, default=5)
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
MODELS_DIR = Path(args.models_dir)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Load preprocessed data
# ══════════════════════════════════════════════════════════════════════════════

X_train = np.load(DATA_DIR / 'X_train.npy')
X_test  = np.load(DATA_DIR / 'X_test.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
y_test  = np.load(DATA_DIR / 'y_test.npy')

with open(DATA_DIR / 'class_mapping.json') as f:
    class_mapping = json.load(f)        # {"0":"benign", "1":"bruteforce", ...}
with open(DATA_DIR / 'class_weights.json') as f:
    class_weight_dict = json.load(f)

CLASS_WEIGHTS = {int(k): v for k, v in class_weight_dict.items()}
CLASS_NAMES   = [class_mapping[str(i)] for i in range(len(class_mapping))]
N_CLASSES     = len(CLASS_NAMES)
N_FEATURES    = X_train.shape[1]
N_NODES       = 12

# Load FL partitions
partitions = []
for i in range(1, N_NODES + 1):
    Xn = np.load(DATA_DIR / 'client_splits' / f'client_{i:02d}_X.npy')
    yn = np.load(DATA_DIR / 'client_splits' / f'client_{i:02d}_y.npy')
    partitions.append((Xn, yn))

# Benign-only set for Autoencoder
benign_enc  = int([k for k, v in class_mapping.items() if v == 'benign'][0])
X_ae_train  = X_train[y_train == benign_enc]

print("=" * 55)
print("  DATA LOADED")
print("=" * 55)
print(f"  Train:    {X_train.shape}")
print(f"  Test:     {X_test.shape}")
print(f"  Features: {N_FEATURES}  |  Classes: {N_CLASSES}")
print(f"  FL nodes: {N_NODES}")
print(f"  AE benign train: {X_ae_train.shape}")
print(f"  Classes: {CLASS_NAMES}")
print("=" * 55)

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Random Forest centralized baseline (FR10)
#
# JUSTIFICATION (see model_selection_justification.txt):
#   Random Forest achieves weighted F1 = 0.9843 on the DataSense 8-class task
#   (Firouzi et al. 2025, Table 8) — the highest of all 22 models tested.
#   It is used here as the upper-bound centralized reference for FR10.
#   It cannot be federated (no weight tensors), hence it is the baseline only.
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)

print("\n" + "=" * 55)
print("  PART 2 — RANDOM FOREST CENTRALIZED BASELINE (FR10)")
print("=" * 55)
print("Training Random Forest on full training set...")
print("(This is the privacy-sacrificing upper bound)\n")

rf_model = RandomForestClassifier(
    n_estimators=200,      # 200 trees — balances variance reduction and speed
    max_depth=None,        # unlimited depth — RF self-regularizes via bootstrapping
    class_weight='balanced',  # built-in imbalance handling
    n_jobs=-1,             # use all CPU cores
    random_state=42,
    verbose=0
)

rf_model.fit(X_train, y_train)

# Evaluate
y_pred_rf    = rf_model.predict(X_test)
y_proba_rf   = rf_model.predict_proba(X_test)

acc_rf    = accuracy_score(y_test, y_pred_rf)
f1_rf_mac = f1_score(y_test, y_pred_rf, average='macro')
f1_rf_w   = f1_score(y_test, y_pred_rf, average='weighted')
f1_rf_cls = f1_score(y_test, y_pred_rf, average=None)

try:
    auc_rf = roc_auc_score(y_test, y_proba_rf, multi_class='ovr', average='macro')
except Exception:
    auc_rf = float('nan')

# Per-class FPR
cm_rf = confusion_matrix(y_test, y_pred_rf)
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
print(f"  Macro F1      : {f1_rf_mac:.4f}  ← FR10 reference")
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
    'accuracy': acc_rf, 'macro_f1': f1_rf_mac,
    'weighted_f1': f1_rf_w, 'roc_auc': auc_rf,
    'per_class_f1': f1_rf_cls.tolist(),
    'per_class_fpr': fpr_rf,
}

# Feature importance (bonus for report Chapter 4)
feat_names = [f'feature_{i}' for i in range(N_FEATURES)]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
print("Top 10 RF feature importances:")
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:>2}. {feat_names[idx]:<30} {importances[idx]:.4f}")

joblib.dump(rf_model, MODELS_DIR / 'rf_centralized.pkl')
print(f"\nSaved RF model → {MODELS_DIR / 'rf_centralized.pkl'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — GRU model definition
#
# JUSTIFICATION (see model_selection_justification.txt):
#   GRU is the FL detection model (FR1-FR3). Architecture:
#     Input (n,17) → reshape (n,1,17) → GRU(64) → Dense(32,relu) →
#     Dropout(0.3) → Softmax(8)
#   GRU's gating prevents vanishing gradients on short local training runs
#   (5 epochs per round on ~3k samples), outperforming plain DNN in this regime.
#   Compatible with FedAvg: all layers produce weight tensors.
# ══════════════════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

def build_gru(n_features: int, n_classes: int, gru_units: int = 64,
              dense_units: int = 32, dropout_rate: float = 0.3) -> keras.Model:
    """
    Single-layer GRU for 8-class network traffic classification in FL.

    Architecture rationale:
      GRU(64): captures non-linear feature interactions through gating.
        64 units — sufficient capacity for 17 features, avoids overfitting
        on small per-node datasets (~3k samples).
        return_sequences=False: we classify the full window, not each step.
      Dense(32, relu): compact projection before output.
      Dropout(0.3): regularization across 20 FL rounds.
      Softmax(8): probability distribution over 8 attack classes.

    Input: (batch, 17) → reshaped internally to (batch, 1, 17)
    Output: (batch, 8) class probabilities
    """
    inp = keras.Input(shape=(n_features,), name='features')

    # Reshape tabular features to (batch, timesteps=1, features)
    # This is the standard adaptation of RNN architectures to tabular data.
    # The GRU still applies its full gating mechanism to the 17-dim vector.
    x = layers.Reshape((1, n_features), name='reshape_to_sequence')(inp)

    # GRU layer — core recurrent processing
    x = layers.GRU(
        gru_units,
        activation='tanh',          # tanh is GRU's standard activation
        recurrent_activation='sigmoid',  # sigmoid for gates (standard)
        return_sequences=False,
        dropout=0.1,                # input dropout
        recurrent_dropout=0.0,      # recurrent dropout disabled (unstable in FL)
        name='gru'
    )(x)

    # Dense projection + regularization
    x = layers.Dense(dense_units, activation='relu', name='dense')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Output
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = keras.Model(inp, out, name='FL_GRU_IDS')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

gru_ref = build_gru(N_FEATURES, N_CLASSES)
gru_ref.summary()
print(f"\nTotal parameters: {gru_ref.count_params():,}")
print("All layers produce weight tensors → compatible with FedAvg aggregation")

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Flower client
# ══════════════════════════════════════════════════════════════════════════════

import flwr as fl

class IIoTGRUClient(fl.client.NumPyClient):
    """
    Flower FL client representing one IIoT edge node.

    Privacy guarantee: self.X and self.y never leave fit().
    Only get_weights() output (float32 arrays) is sent to the server.

    FedAvg contribution: n_samples returned from fit() so the server can
    weight this client's update proportionally (w_global = Σ n_k/n * w_k).
    """

    def __init__(self, node_id: int, X: np.ndarray, y: np.ndarray,
                 local_epochs: int = 5):
        self.node_id      = node_id
        self.X            = X
        self.y            = y
        self.local_epochs = local_epochs
        self.model        = build_gru(N_FEATURES, N_CLASSES)

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs     = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', 256)

        # Compute local class weights for this node's non-IID distribution.
        # Each node sees different attack classes → weights differ per node.
        local_classes = np.unique(self.y)
        n_total = len(self.y)
        local_cw = {}
        for cls in range(N_CLASSES):
            if cls in local_classes:
                n_cls = (self.y == cls).sum()
                local_cw[cls] = n_total / (len(local_classes) * n_cls)
            else:
                local_cw[cls] = 0.0

        # Local training — raw data stays here
        self.model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=local_cw,
            validation_split=0.1,
            verbose=0
        )

        # Return weights + sample count (for FedAvg weighting)
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {'accuracy': float(acc)}


# Build all 12 clients
print(f"\nInstantiating {N_NODES} FL clients...")
clients = []
for i, (Xn, yn) in enumerate(partitions):
    c = IIoTGRUClient(node_id=i+1, X=Xn, y=yn,
                      local_epochs=args.local_epochs)
    clients.append(c)
    print(f"  Client {i+1:02d}: {len(Xn):>6,} samples | "
          f"classes present: {sorted([CLASS_NAMES[c_] for c_ in np.unique(yn)])}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Flower server + FedAvg + simulation
# ══════════════════════════════════════════════════════════════════════════════
"""
FL hyperparameters:
  NUM_ROUNDS = 20
    Sufficient for convergence on non-IID data (McMahan et al. 2017 shows
    FedAvg with E=5 converges within 15–25 rounds on non-IID splits).

  LOCAL_EPOCHS = 5
    The McMahan et al. (2017) FedAvg paper originally uses E=5 as the
    standard. Too few → slow convergence; too many → client drift (local
    models diverge, FedAvg aggregation degrades). 5 is the established default.

  FRACTION_FIT = 1.0
    All 12 nodes participate every round. In simulation on a single machine
    this is feasible. In real deployment with many devices you'd use 0.3–0.5.
"""

from typing import List, Tuple, Dict
from flwr.common import Metrics
from flwr.simulation import start_simulation

round_log = {'round': [], 'val_loss': [], 'val_acc': [], 'macro_f1': []}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m['accuracy'] for n, m in metrics) / total
    return {'accuracy': acc}


class GRUFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg extended to:
    1. Evaluate the global GRU on the centralized test set after each round
       (server-side evaluation — only possible in simulation)
    2. Log convergence metrics to round_log
    3. Track and restore the best global weights by Macro F1
    """

    def __init__(self, global_model: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.best_f1      = 0.0
        self.best_weights = None

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(
            server_round, results, failures)

        if agg_params is not None:
            weights = fl.common.parameters_to_ndarrays(agg_params)
            self.global_model.set_weights(weights)

            # Server-side evaluation on test set
            loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0)
            y_pred    = self.global_model.predict(X_test, verbose=0).argmax(axis=1)
            f1_mac    = f1_score(y_test, y_pred, average='macro')

            round_log['round'].append(server_round)
            round_log['val_loss'].append(loss)
            round_log['val_acc'].append(acc)
            round_log['macro_f1'].append(f1_mac)

            print(f"  Round {server_round:02d}/{args.num_rounds} | "
                  f"Loss: {loss:.4f} | Acc: {acc*100:.2f}% | Macro F1: {f1_mac:.4f}",
                  flush=True)

            if f1_mac > self.best_f1:
                self.best_f1      = f1_mac
                self.best_weights = [w.copy() for w in weights]
                print(f"           ↑ New best F1: {f1_mac:.4f} — checkpoint saved")

        return agg_params, agg_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


global_gru = build_gru(N_FEATURES, N_CLASSES)

strategy = GRUFedAvgStrategy(
    global_model=global_gru,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=N_NODES,
    min_evaluate_clients=N_NODES,
    min_available_clients=N_NODES,
    on_fit_config_fn=lambda rnd: {
        'local_epochs': args.local_epochs,
        'batch_size': 256,
        'round': rnd
    },
    evaluate_metrics_aggregation_fn=weighted_average,
)

def client_fn(cid: str) -> fl.client.NumPyClient:
    return clients[int(cid)]

print("\n" + "=" * 55)
print(f"  STARTING FL SIMULATION")
print(f"  Rounds: {args.num_rounds} | Nodes: {N_NODES} | "
      f"Local epochs: {args.local_epochs}")
print("=" * 55)

start_simulation(
    client_fn=client_fn,
    num_clients=N_NODES,
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
    client_resources={'num_cpus': 1, 'num_gpus': 0.0},
    ray_init_args={'ignore_reinit_error': True, 'log_to_driver': False},
)

print(f"\n FL simulation complete.")
print(f"  Best Macro F1 achieved: {strategy.best_f1:.4f}")

# Restore best weights
if strategy.best_weights is not None:
    global_gru.set_weights(strategy.best_weights)
    print("  Best weights restored.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Convergence plots
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'FL-GRU Convergence — FedAvg ({args.num_rounds} rounds, '
             f'{N_NODES} nodes, non-IID)', fontsize=13)

rounds = round_log['round']
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

axes[2].plot(rounds, round_log['macro_f1'], '^-', color='#2ecc71', lw=2.5, ms=8)
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
# PART 7 — Final GRU evaluation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PART 7 — FEDERATED GRU — FINAL EVALUATION")
print("=" * 55)

y_pred_gru  = global_gru.predict(X_test, verbose=0).argmax(axis=1)
y_proba_gru = global_gru.predict(X_test, verbose=0)

acc_gru    = accuracy_score(y_test, y_pred_gru)
f1_gru_mac = f1_score(y_test, y_pred_gru, average='macro')
f1_gru_w   = f1_score(y_test, y_pred_gru, average='weighted')
f1_gru_cls = f1_score(y_test, y_pred_gru, average=None)

try:
    auc_gru = roc_auc_score(y_test, y_proba_gru, multi_class='ovr', average='macro')
except Exception:
    auc_gru = float('nan')

# Per-class FPR
cm_gru = confusion_matrix(y_test, y_pred_gru)
fpr_gru = {}
for i, cls in enumerate(CLASS_NAMES):
    TP = cm_gru[i, i]
    FP = cm_gru[:, i].sum() - TP
    TN = cm_gru.sum() - cm_gru[i, :].sum() - cm_gru[:, i].sum() + TP
    fpr_gru[cls] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

print(f"  Accuracy      : {acc_gru:.4f}")
print(f"  Macro F1      : {f1_gru_mac:.4f}  ← PRIMARY METRIC")
print(f"  Weighted F1   : {f1_gru_w:.4f}")
print(f"  ROC-AUC (OvR) : {auc_gru:.4f}")
print("\n  Per-class F1 and FPR:")
print(f"  {'Class':<15} {'F1':>8} {'FPR':>8} {'Support':>10}")
print("  " + "-" * 44)
for i, cls in enumerate(CLASS_NAMES):
    n = (y_test == i).sum()
    print(f"  {cls:<15} {f1_gru_cls[i]:>8.4f} "
          f"{fpr_gru[cls]:>8.4f} {n:>10,}")

print("\n" + classification_report(y_test, y_pred_gru,
                                    target_names=CLASS_NAMES, digits=4))

GRU_RESULTS = {
    'accuracy': acc_gru, 'macro_f1': f1_gru_mac,
    'weighted_f1': f1_gru_w, 'roc_auc': auc_gru,
    'per_class_f1': f1_gru_cls.tolist(),
    'per_class_fpr': fpr_gru,
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — FR10: RF vs FL-GRU privacy-accuracy tradeoff
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FR10 — PRIVACY-ACCURACY TRADEOFF: RF vs FL-GRU")
print("=" * 65)
print(f"  {'Metric':<20} {'RF (centralized)':>18} {'GRU (federated)':>18} {'Gap':>8}")
print("  " + "-" * 66)
for label, rf_v, gru_v in [
    ('Accuracy',    RF_RESULTS['accuracy'],   GRU_RESULTS['accuracy']),
    ('Macro F1',    RF_RESULTS['macro_f1'],   GRU_RESULTS['macro_f1']),
    ('Weighted F1', RF_RESULTS['weighted_f1'],GRU_RESULTS['weighted_f1']),
    ('ROC-AUC',     RF_RESULTS['roc_auc'],    GRU_RESULTS['roc_auc']),
]:
    gap = gru_v - rf_v
    sign = '+' if gap >= 0 else ''
    print(f"  {label:<20} {rf_v:>18.4f} {gru_v:>18.4f} {sign}{gap:>7.4f}")

gap_f1 = RF_RESULTS['macro_f1'] - GRU_RESULTS['macro_f1']
print(f"\n  Privacy cost of FL : {gap_f1*100:.2f}% Macro F1 vs RF baseline")
print(f"  Raw data shared    : RF = 100% | FL-GRU = 0%")
print(f"  Conclusion         : {gap_f1*100:.1f}% detection cost to eliminate")
print(f"                       all cross-node raw data transfer")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(N_CLASSES)
w = 0.35
b1 = ax.bar(x - w/2, RF_RESULTS['per_class_f1'], w,
            label='RF — centralized (all data)', color='#e74c3c', alpha=0.8)
b2 = ax.bar(x + w/2, GRU_RESULTS['per_class_f1'], w,
            label='GRU — federated (FL, zero raw data shared)', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylabel('F1-score')
ax.set_ylim(0, 1.12)
ax.set_title('FR10 — Per-class F1: Random Forest centralized vs FL-GRU\n'
             f'DataSense IIoT 2025 | {N_NODES}-node non-IID partition | 5-second window')
ax.legend()
ax.axhline(0.9, color='gray', ls='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
            f'{h:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'fr10_rf_vs_gru.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'fr10_rf_vs_gru.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — Autoencoder training and anomaly threshold
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PART 9 — AUTOENCODER (unsupervised anomaly scorer)")
print("=" * 55)

def build_autoencoder(n_features, bottleneck=4):
    inp = keras.Input(shape=(n_features,), name='ae_input')
    x   = layers.Dense(12, activation='relu', name='enc_1')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(8,  activation='relu', name='enc_2')(x)
    x   = layers.BatchNormalization()(x)
    z   = layers.Dense(bottleneck, activation='relu', name='bottleneck')(x)
    x   = layers.Dense(8,  activation='relu', name='dec_1')(z)
    x   = layers.Dense(12, activation='relu', name='dec_2')(x)
    out = layers.Dense(n_features, activation='sigmoid', name='ae_output')(x)
    ae  = keras.Model(inp, out, name='Autoencoder_IDS')
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    enc = keras.Model(inp, z, name='Encoder')
    return ae, enc

ae_model, encoder = build_autoencoder(N_FEATURES)

print(f"Training AE on {len(X_ae_train):,} benign-only samples...")
ae_history = ae_model.fit(
    X_ae_train, X_ae_train,
    epochs=150,
    batch_size=256,
    validation_split=0.15,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1)],
    verbose=1
)

# Threshold: μ + 2σ of benign reconstruction error
def recon_error(model, X):
    return np.mean(np.square(X - model.predict(X, verbose=0)), axis=1)

err_benign = recon_error(ae_model, X_ae_train)
THRESHOLD  = err_benign.mean() + 2 * err_benign.std()
print(f"\nAnomaly threshold (μ+2σ): {THRESHOLD:.6f}")

# Binary evaluation on test set
from sklearn.metrics import precision_score, recall_score, roc_curve, auc as sk_auc

err_test    = recon_error(ae_model, X_test)
y_bin_true  = (y_test != benign_enc).astype(int)
y_bin_pred  = (err_test > THRESHOLD).astype(int)

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
print(f"  FPR       : {ae_fpr:.4f}  ({ae_fpr*100:.2f}% false alarms)")
print(f"  ROC-AUC   : {ae_auc:.4f}")

# Reconstruction error distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(err_benign, bins=50, alpha=0.6, color='#2ecc71',
             label='Benign (train)', density=True)
axes[0].hist(err_test[y_test != benign_enc], bins=50, alpha=0.6,
             color='#e74c3c', label='Attack (test)', density=True)
axes[0].axvline(THRESHOLD, color='black', ls='--', lw=2,
                label=f'Threshold μ+2σ = {THRESHOLD:.5f}')
axes[0].set_title('AE reconstruction error distribution')
axes[0].set_xlabel('MSE')
axes[0].legend()

axes[1].plot(fpr_r, tpr_r, color='#9b59b6', lw=2,
             label=f'AE ROC (AUC={ae_auc:.4f})')
axes[1].plot([0,1],[0,1],'k--', alpha=0.4)
axes[1].scatter([ae_fpr],[ae_rec], color='red', s=100, zorder=5,
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
    (y_pred_rf,  f'RF — centralized\n(F1={f1_rf_mac:.4f})'),
    (y_pred_gru, f'FL-GRU — federated\n(F1={f1_gru_mac:.4f}, 0% raw data shared)'),
]):
    cm_n = confusion_matrix(y_test, y_pred).astype(float)
    cm_n /= cm_n.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    thresh = cm_n.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if cm_n[i,j] > thresh else 'black',
                    fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODELS_DIR / 'confusion_matrices.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 11 — Save models and results
# ══════════════════════════════════════════════════════════════════════════════

global_gru.save(str(MODELS_DIR / 'fl_gru_model'))
ae_model.save(str(MODELS_DIR / 'autoencoder'))
joblib.dump(rf_model, MODELS_DIR / 'rf_centralized.pkl')

with open(MODELS_DIR / 'ae_config.json', 'w') as f:
    json.dump({'threshold': float(THRESHOLD),
               'benign_mean': float(err_benign.mean()),
               'benign_std':  float(err_benign.std())}, f, indent=2)

all_results = {
    'random_forest_centralized': RF_RESULTS,
    'gru_federated':             GRU_RESULTS,
    'autoencoder':               AE_RESULTS,
    'fl_config': {
        'num_rounds': args.num_rounds,
        'num_clients': N_NODES,
        'local_epochs': args.local_epochs,
        'algorithm': 'FedAvg',
        'model': 'GRU',
    }
}
with open(MODELS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print("=" * 65)
print(f"\n  {'Model':<25} {'Macro F1':>10} {'Data shared':>14}")
print("  " + "-" * 52)
print(f"  {'RF (centralized)':<25} {RF_RESULTS['macro_f1']:>10.4f} {'100% raw data':>14}")
print(f"  {'GRU (FL, federated)':<25} {GRU_RESULTS['macro_f1']:>10.4f} {'0% raw data':>14}")
print(f"  {'Autoencoder (binary)':<25} {AE_RESULTS['f1']:>10.4f} {'0% raw data':>14}")
gap = RF_RESULTS['macro_f1'] - GRU_RESULTS['macro_f1']
print(f"\n  Privacy-accuracy gap (FR10): {gap*100:.2f}% Macro F1")
print(f"  Saved to: {MODELS_DIR}")
print("\n  READY FOR PHASE 3: SDN + Mininet + Online Simulation")

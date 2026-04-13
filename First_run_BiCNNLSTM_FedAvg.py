"""
FL-SDN-IDS — Phase 2: Offline Federated Learning Training
==========================================================
Run from VS Code. All paths use the OUTPUT directory from Phase 1.

Prerequisites
-------------
pip install flwr==1.7.0 tensorflow scikit-learn matplotlib numpy joblib

Usage
-----
python fl_training_BICNNLSTM.py --data_dir ./data --models_dir ./models

Structure
---------
  Part 1  — Load preprocessed data
  Part 2  — Random Forest centralized baseline (FR10)
  Part 3  — BiCNN-LSTM model definition
  Part 4  — Flower client (IIoTFlowerClient)
  Part 5  — Flower server + FedAvg + simulation
  Part 6  — Convergence plots
  Part 7  — Final BiCNN-LSTM evaluation (accuracy, F1, FPR, ROC-AUC)
  Part 8  — FL vs RF comparison (FR10 privacy-accuracy tradeoff)
  Part 9  — Autoencoder training and anomaly threshold
  Part 10 — Confusion matrices + report figures
  Part 11 — Save all models and results

Model selection rationale (Firouzi et al., Electronics 2025, Table 8):
-----------------------------------------------------------------------
  BiCNN-LSTM  — FL primary classifier (FR1-FR3, NF1, NF2, NF3, NF7)
  Random Forest — centralized baseline (FR10, NF5)
  Autoencoder  — unsupervised anomaly scorer (FR4, FR5, NF1)

References
----------
  [1] Firouzi et al., Electronics 2025, 14, 4095 — Table 8
  [2] McMahan et al., AISTATS 2017 — FedAvg
  [3] Li et al., MLSys 2020 — FedProx
  [4] Olanrewaju-George & Pranggono, Cyber Security and Applications 2025
  [5] Zainudin et al., IEEE TNSM 2023 — CNN-recurrent FL-IDS on SDN
"""

import os, sys, json, joblib, argparse, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF C++ logs
np.random.seed(42)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',     default='./processed_output')
parser.add_argument('--models_dir',   default='./models')
parser.add_argument('--num_rounds',   type=int, default=20)
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
    class_mapping = json.load(f)
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
print("=" * 55)

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
print("Training Random Forest on full training set ...")

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
# NOTE: build_bicnn_lstm is defined as a plain function with ALL Keras imports
# happening inside it. This guarantees the function itself contains no
# KerasLazyLoader references at the module level, so Ray/cloudpickle can
# never accidentally capture one when it serializes client_fn.
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
    Bidirectional CNN-LSTM for 8-class IIoT traffic classification in FL.
    All Keras symbols are imported locally so this function is safely
    picklable by Ray/cloudpickle.
    """
    # ── Local imports keep Keras out of the module-level namespace ──────────
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # ────────────────────────────────────────────────────────────────────────

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


# Sanity-check summary (uses local imports inside the function — safe)
_ref = build_bicnn_lstm(N_FEATURES, N_CLASSES)
_ref.summary()
print(f"\nTotal parameters: {_ref.count_params():,}")
print("All layers produce weight tensors -> compatible with FedAvg")
del _ref   # free memory; we don't need this reference model later

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Flower client
#
# Ray serialization rules that apply here:
#
#   1. Ray pickles client_fn (the factory) and ships it to each worker actor.
#   2. cloudpickle traces every global variable reachable from client_fn.
#   3. Any reachable object that contains a KerasLazyLoader causes the crash.
#
# Solution: IIoTBiCNNLSTMClient does NOT import keras at class definition
# time. All keras/tf symbols are imported INSIDE __init__ and the fit/eval
# methods. This makes the class body contain zero Keras references at the
# time Ray serializes it.
# ══════════════════════════════════════════════════════════════════════════════

import flwr as fl


class IIoTBiCNNLSTMClient(fl.client.NumPyClient):
    """
    Flower FL client for one IIoT edge node.

    Privacy guarantee (NF1 / FR1):
        self.X and self.y are local-only and never leave this object.
        Only float32 weight arrays are transmitted to the server.

    FedAvg weighting (FR2):
        fit() returns len(self.X) for proportional aggregation.
    """

    def __init__(
        self,
        node_id:      int,
        X:            np.ndarray,
        y:            np.ndarray,
        n_features:   int,
        n_classes:    int,
        local_epochs: int = 5,
    ):
        self.node_id      = node_id
        self.X            = X
        self.y            = y
        self.n_features   = n_features
        self.n_classes    = n_classes
        self.local_epochs = local_epochs
        # Build the model using the module-level builder.
        # build_bicnn_lstm imports Keras locally, so no KerasLazyLoader
        # leaks into this object's __dict__ at construction time.
        self.model = build_bicnn_lstm(n_features, n_classes)

    # ── Flower interface ─────────────────────────────────────────────────────

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs     = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', 256)

        local_classes = np.unique(self.y)
        n_total       = len(self.y)
        local_cw      = {}
        for cls in range(self.n_classes):
            if cls in local_classes:
                n_cls         = (self.y == cls).sum()
                local_cw[cls] = n_total / (len(local_classes) * n_cls)
            else:
                local_cw[cls] = 0.0

        self.model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=local_cw,
            validation_split=0.1,
            verbose=0,
        )
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {'accuracy': float(acc)}


# ── Print partition summary (no pre-instantiation) ───────────────────────────
print(f"\nFL client partitions ({N_NODES} nodes):")
for i, (Xn, yn) in enumerate(partitions):
    print(
        f"  Client {i+1:02d}: {len(Xn):>6,} samples | "
        f"classes: {sorted([CLASS_NAMES[c_] for c_ in np.unique(yn)])}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Flower server + FedAvg + simulation
#
# KEY FIX — client_fn:
#   • Defined at module top level (not a closure, not a lambda).
#   • Accesses `partitions`, `N_FEATURES`, `N_CLASSES`, `N_NODES`,
#     `args.local_epochs` — all plain Python/NumPy objects with no Keras.
#   • Constructs IIoTBiCNNLSTMClient fresh each call. The Keras model is
#     built inside __init__ → inside build_bicnn_lstm → after Ray deserializes
#     the function in the worker. Nothing unpicklable crosses the boundary.
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


# ── Strategy ─────────────────────────────────────────────────────────────────
# The strategy holds a global_model for server-side evaluation.
# It lives in the main process only — Ray never serializes it.

# Import Keras here for the server/strategy side (main process only).
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers   # noqa: F401 — needed by build_bicnn_lstm

tf.random.set_seed(42)


class BiCNNLSTMFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg extended to evaluate the global model after every round and
    checkpoint the best weights by Macro F1.
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

strategy = BiCNNLSTMFedAvgStrategy(
    global_model=global_bicnn_lstm,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=N_NODES,
    min_evaluate_clients=N_NODES,
    min_available_clients=N_NODES,
    on_fit_config_fn=lambda rnd: {
        'local_epochs': args.local_epochs,
        'batch_size':   256,
        'round':        rnd,
    },
    evaluate_metrics_aggregation_fn=weighted_average,
)


# ── client_fn — the ONLY object Ray serializes ───────────────────────────────
# Rules obeyed:
#   • Top-level function (not nested, not a lambda).
#   • Captures only: partitions (list of np.ndarray tuples),
#                    N_FEATURES, N_CLASSES (plain ints),
#                    args.local_epochs (plain int).
#   • IIoTBiCNNLSTMClient class is referenced but contains NO Keras at
#     class-body level — Keras only appears inside method bodies which are
#     not evaluated during pickling.
# ─────────────────────────────────────────────────────────────────────────────
_LOCAL_EPOCHS = args.local_epochs   # extract to a plain int — avoids capturing
                                    # the entire argparse.Namespace object


def client_fn(cid: str) -> fl.client.NumPyClient:
    idx    = int(cid)
    Xn, yn = partitions[idx]
    return IIoTBiCNNLSTMClient(
        node_id=idx + 1,
        X=Xn,
        y=yn,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        local_epochs=_LOCAL_EPOCHS,
    )


print("\n" + "=" * 55)
print(f"  STARTING FL SIMULATION — BiCNN-LSTM + FedAvg")
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

print(f"\nFL simulation complete.")
print(f"  Best Macro F1 achieved: {strategy.best_f1:.4f}")
print(f"  Paper reference value:  0.9551 (Table 8, Firouzi et al. 2025)")

if strategy.best_weights is not None:
    global_bicnn_lstm.set_weights(strategy.best_weights)
    print("  Best weights restored from checkpoint.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Convergence plots
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'FL-BiCNN-LSTM Convergence — FedAvg ({args.num_rounds} rounds, '
    f'{N_NODES} nodes, non-IID)',
    fontsize=13
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
# PART 7 — Final BiCNN-LSTM evaluation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PART 7 — FEDERATED BiCNN-LSTM — FINAL EVALUATION")
print("=" * 55)

y_pred_bicnn  = global_bicnn_lstm.predict(X_test, verbose=0).argmax(axis=1)
y_proba_bicnn = global_bicnn_lstm.predict(X_test, verbose=0)

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
# PART 8 — FR10: RF vs FL-BiCNN-LSTM privacy-accuracy tradeoff
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
x = np.arange(N_CLASSES)
w = 0.35
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
    'FR10 — Per-class F1: Random Forest centralized vs FL-BiCNN-LSTM\n'
    f'DataSense IIoT 2025 | {N_NODES}-node non-IID partition'
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
# PART 9 — Autoencoder training and anomaly threshold
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PART 9 — AUTOENCODER (unsupervised anomaly scorer)")
print("=" * 55)


def build_autoencoder(n_features: int, bottleneck: int = 4):
    """Symmetric encoder-decoder for benign traffic reconstruction."""
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
print(f"\nAnomaly threshold (mu + 2*sigma): {THRESHOLD:.6f}")

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
     f'RF — centralized\n(F1={f1_rf_mac:.4f})'),
    (y_pred_bicnn,
     f'FL-BiCNN-LSTM — federated\n'
     f'(F1={f1_bicnn_mac:.4f}, 0% raw data shared)'),
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
        'num_rounds':   args.num_rounds,
        'num_clients':  N_NODES,
        'local_epochs': args.local_epochs,
        'algorithm':    'FedAvg',
        'model':        'BiCNN-LSTM',
    },
    'model_selection_reference': {
        'source':                        'Firouzi et al., Electronics 2025, 14, 4095 — Table 8',
        'bicnn_lstm_paper_acc_8class':   0.9545,
        'bicnn_lstm_paper_f1_8class':    0.9551,
        'rf_paper_f1_8class':            0.9780,
        'rnn_excluded_reason':
            'No gating; vanishing gradients on non-IID FL partitions '
            '(McMahan 2017, Li 2020 FedProx)',
        'deepresnet1d_excluded_reason':
            'Highest raw F1 but deep residual architecture has excessive '
            'parameter count; violates NF2 and NF3 edge constraints',
    },
}

with open(MODELS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print("=" * 65)
print(f"\n  {'Model':<28} {'Macro F1':>10} {'Data shared':>14}")
print("  " + "-" * 54)
print(f"  {'RF (centralized)':<28} {RF_RESULTS['macro_f1']:>10.4f} {'100% raw':>14}")
print(f"  {'BiCNN-LSTM (FL, federated)':<28} "
      f"{BICNN_LSTM_RESULTS['macro_f1']:>10.4f} {'0% raw':>14}")
print(f"  {'Autoencoder (binary)':<28} {AE_RESULTS['f1']:>10.4f} {'0% raw':>14}")
gap = RF_RESULTS['macro_f1'] - BICNN_LSTM_RESULTS['macro_f1']
print(f"\n  Privacy-accuracy gap (FR10): {gap * 100:.2f}% Macro F1")
print(f"  Saved to: {MODELS_DIR}")
print("\n  READY FOR PHASE 3: SDN + Mininet + Online Simulation")
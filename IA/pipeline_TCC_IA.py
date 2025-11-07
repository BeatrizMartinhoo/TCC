#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG – Arritmias (DII Único) — Pipeline Consolidado em Python

Descrição:
    Pipeline para análise e classificação de arritmias utilizando apenas
    a derivação DII do dataset UCI Arrhythmia. Inclui:
        A) Pré-processamento e seleção de features (globais + DII específicas)
        B) Avaliação por **100 execuções** (SVM, RF e HGB) com métricas por execução (média ± desvio)
        C) (opcional) MLP (Keras) para dados tabulares
        D) (opcional) Simulação DWT (mantido para uso futuro)

Autora: Beatriz Luísie
Projeto: TCC – Wearable de ECG para Identificação de Arritmias
Instituição: Pontifícia Universidade Católica de São Paulo (PUC-SP)
Ano: 2025
"""

# ==== Utilidades base ====
import os
import sys
import json
import math
import time
import random
from pathlib import Path

# ==== NumPy / Pandas / Plot ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Scikit-learn ====
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, accuracy_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight

# ==== Persistência (se quiser salvar depois) ====
import joblib

# ==== I/O remoto e transformadas (mantido p/ DWT futuro) ====
import requests
import pywt  # opcional

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
HEALTHY_CLASS = 1
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)

RUN_TABULAR    = True
RUN_SIM_DWT    = False
RUN_DL_TABULAR = False
RUN_DL_RAW     = False

# -------------------------------------------------------------
# Aquisição/feature set — Somente DII
# -------------------------------------------------------------
FEATURE_IDS_1BASED = [
    # Globais
    5, 6, 7, 8, 9, 15,
    # DII — larguras / flags
    28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39,
    # DII — amplitudes e áreas
    170, 171, 172, 173, 174, 175, 176, 177,
    178, 179
]
CATEGORICAL_IDS_1BASED = [34, 35, 36, 37, 38, 39]
NUMERIC_IDS_1BASED = [i for i in FEATURE_IDS_1BASED if i not in CATEGORICAL_IDS_1BASED]

FEATURE_COLS     = [i - 1 for i in FEATURE_IDS_1BASED]
CATEGORICAL_COLS = [i - 1 for i in CATEGORICAL_IDS_1BASED]
NUMERIC_COLS     = [i - 1 for i in NUMERIC_IDS_1BASED]
AMPL_COLS_01mV   = [c - 1 for c in [170,171,172,173,174,175,176,177]]

CONVERT_AMPL_TO_MV = True
MV_SCALE = 0.1  # 0,1 mV → mV

TEST_SIZE = 0.20
VAL_SIZE  = 0.20

# ============================================================
# BLOCO 2 — LEITURA E PRÉ-PROCESSAMENTO (DII)
# ============================================================
from typing import Tuple, Dict, Any
from sklearn.impute import SimpleImputer

URL_DATASET = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"

def carregar_dataset(caminho_csv: str = URL_DATASET, sep: str = ",") -> pd.DataFrame:
    if caminho_csv.startswith("http"):
        print("🔗 Baixando dataset diretamente da UCI...\n")
        df = pd.read_csv(
            caminho_csv, sep=sep,
            na_values=["?", "NA", "NaN", "nan", ""],
            engine="python"
        )
    else:
        p = Path(caminho_csv)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        df = pd.read_csv(
            p, sep=sep,
            na_values=["?", "NA", "NaN", "nan", ""],
            engine="python"
        )
    df.columns = [f"f{i+1}" for i in range(df.shape[1])]
    return df

def _detectar_coluna_alvo(df: pd.DataFrame) -> str:
    candidatos = ["class", "Class", "target", "Target", "arrhythmia", "Arrhythmia"]
    for c in candidatos:
        if c in df.columns:
            return c
    return df.columns[-1]

def selecionar_colunas_DII(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    alvo = _detectar_coluna_alvo(df)
    X = df.iloc[:, FEATURE_COLS].copy()
    y = df[alvo].copy()

    subset_pos = {orig_idx: pos for pos, orig_idx in enumerate(FEATURE_COLS)}
    cat_idx_rel = [subset_pos[i] for i in CATEGORICAL_COLS if i in subset_pos]
    num_idx_rel = [subset_pos[i] for i in NUMERIC_COLS if i in subset_pos]
    amp_idx_rel = [subset_pos[i] for i in AMPL_COLS_01mV if i in subset_pos]

    meta = {
        "alvo": alvo,
        "categorical_cols_rel": cat_idx_rel,
        "numeric_cols_rel": num_idx_rel,
        "amplitude_cols_rel": amp_idx_rel,
        "convert_ampl_to_mv": CONVERT_AMPL_TO_MV,
        "mv_scale": MV_SCALE,
    }
    return X, y, meta

def preprocessar_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    meta: Dict[str, Any],
    padronizar: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    X = X.copy()
    cat_idx = meta.get("categorical_cols_rel", [])
    num_idx = meta.get("numeric_cols_rel", [])
    amp_idx = meta.get("amplitude_cols_rel", [])
    conv_amp = meta["convert_ampl_to_mv"]
    mv_scale = meta["mv_scale"]

    # 1) Converter amplitudes 0,1 mV → mV
    if conv_amp and len(amp_idx) > 0:
        valid_idx = [i for i in amp_idx if i < X.shape[1]]
        if valid_idx:
            inter = X.iloc[:, valid_idx].astype(float)
            X.iloc[:, valid_idx] = inter * mv_scale

    # 2) Imputações
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")

    X_num = X.iloc[:, num_idx].apply(pd.to_numeric, errors="coerce")
    X_cat = X.iloc[:, cat_idx].copy()
    for c in X_cat.columns:
        X_cat[c] = pd.to_numeric(X_cat[c], errors="coerce")

    X_num_imp = pd.DataFrame(num_imp.fit_transform(X_num), columns=X_num.columns, index=X.index)
    X_cat_imp = pd.DataFrame(cat_imp.fit_transform(X_cat), columns=X_cat.columns, index=X.index)
    X_cat_imp = X_cat_imp.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0).astype(int)

    # 3) Padronização
    if padronizar and len(num_idx) > 0:
        scaler = StandardScaler()
        X_num_std = pd.DataFrame(
            scaler.fit_transform(X_num_imp),
            columns=X_num_imp.columns, index=X.index
        )
    else:
        scaler = None
        X_num_std = X_num_imp

    # 4) Reconstrução preservando ordem
    cols_order = list(range(X.shape[1]))
    partes = []
    for j in cols_order:
        colname = X.columns[j]
        if j in num_idx:
            partes.append(X_num_std[[colname]])
        else:
            partes.append(X_cat_imp[[colname]].astype(int))
    X_pre = pd.concat(partes, axis=1)

    artefatos = {
        "num_imputer": num_imp,
        "cat_imputer": cat_imp,
        "scaler": scaler,
    }
    return X_pre, y.copy(), artefatos

def dividir_conjuntos(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = SEED,
):
    X_trval, X_te, y_trval, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=val_size, stratify=y_trval, random_state=random_state
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te

def to_binary_labels(y: pd.Series, healthy_class: int = HEALTHY_CLASS) -> pd.Series:
    yb = (y.astype(int) != healthy_class).astype(int)
    yb.name = "arrhythmia_bin"
    return yb

# ============================================================
# BLOCO 3 — MULTI-RUNS (100×) E AVALIAÇÃO DE ENSEMBLE
# ============================================================
def avaliar(y_true, y_pred, titulo="", y_score=None):
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_mic = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print(f"\n=== {titulo} ===")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1_mac:.4f} | F1-micro: {f1_mic:.4f}")

    if set(np.unique(y_true)) <= {0, 1} and y_score is not None:
        try:
            roc = roc_auc_score(y_true, y_score)
            pr  = average_precision_score(y_true, y_score)
            print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
        except Exception:
            pass

    print("\nRelatório por classe:")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    return {"acc": acc, "f1_macro": f1_mac, "f1_micro": f1_mic}

def svm_multi_runs_probs(X_tr, y_tr, X_te, n_runs=100, C=2.0, gamma="scale", kernel="rbf"):
    proba_mat = np.zeros((X_te.shape[0], n_runs), dtype=float)
    for k in range(n_runs):
        svm = SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced",
                  probability=True, random_state=SEED + k)
        svm.fit(X_tr, y_tr)
        proba_mat[:, k] = svm.predict_proba(X_te)[:, 1]
    return proba_mat

def rf_multi_runs_probs(X_tr, y_tr, X_te, n_runs=100,
                        n_estimators=400, max_depth=None, min_samples_leaf=2):
    proba_mat = np.zeros((X_te.shape[0], n_runs), dtype=float)
    for k in range(n_runs):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED + k
        )
        rf.fit(X_tr, y_tr)
        proba_mat[:, k] = rf.predict_proba(X_te)[:, 1]
    return proba_mat

def hgb_multi_runs_probs(X_tr, y_tr, X_te, n_runs=100,
                         learning_rate=0.06, max_leaf_nodes=31,
                         min_samples_leaf=20, l2_regularization=1e-3):
    """
    Treina HGB 100x mudando random_state e retorna matriz (n_test, n_runs)
    com as probabilidades da classe positiva. Lida com desbalanceamento via sample_weight.
    """
    proba_mat = np.zeros((X_te.shape[0], n_runs), dtype=float)
    sw = compute_sample_weight(class_weight="balanced", y=y_tr)  # pesos por amostra
    for k in range(n_runs):
        rs = SEED + k
        hgb = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=rs
        )
        hgb.fit(X_tr, y_tr, sample_weight=sw)
        proba_mat[:, k] = hgb.predict_proba(X_te)[:, 1]
    return proba_mat

def avaliar_ensemble_from_probas(y_true, proba_mat, titulo, threshold=0.5):
    """
    Calcula métricas a partir do ensemble (média das probabilidades nas execuções)
    e retorna dicionário de métricas + vetores úteis.
    """
    mean_proba = proba_mat.mean(axis=1)
    y_pred = (mean_proba >= threshold).astype(int)
    res = avaliar(y_true, y_pred, titulo=titulo, y_score=mean_proba)
    return res, y_pred, mean_proba

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def avaliar_100_execucoes(y_true, proba_mat, titulo, threshold=0.5):
    """
    y_true: vetor binário (0/1) do conjunto de teste
    proba_mat: [n_amostras_teste, n_runs] com P(y=1) de cada execução
    Imprime e retorna médias ± desvios de Accuracy, F1-macro, ROC-AUC, PR-AUC.
    """
    accs, f1ms, rocs, prs = [], [], [], []
    for k in range(proba_mat.shape[1]):
        p = proba_mat[:, k]
        y_pred = (p >= threshold).astype(int)
        accs.append(accuracy_score(y_true, y_pred))
        f1ms.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        try:
            rocs.append(roc_auc_score(y_true, p))
            prs.append(average_precision_score(y_true, p))
        except Exception:
            pass

    def m_sd(v):
        v = np.array(v, dtype=float)
        if v.size == 0:
            return np.nan, np.nan
        return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else (float(v.mean()), 0.0)

    acc_m, acc_sd = m_sd(accs)
    f1_m,  f1_sd  = m_sd(f1ms)
    roc_m, roc_sd = m_sd(rocs)
    pr_m,  pr_sd  = m_sd(prs)

    print(f"\n=== {titulo} — 100 execuções ===")
    print(f"Accuracy : {acc_m:.4f} ± {acc_sd:.4f}")
    print(f"F1-macro: {f1_m:.4f} ± {f1_sd:.4f}")
    if not np.isnan(roc_m): print(f"ROC-AUC : {roc_m:.4f} ± {roc_sd:.4f}")
    if not np.isnan(pr_m):  print(f"PR-AUC  : {pr_m:.4f} ± {pr_sd:.4f}")

    return {
        "acc_mean": acc_m, "acc_sd": acc_sd,
        "f1m_mean": f1_m,  "f1m_sd": f1_sd,
        "roc_mean": roc_m, "roc_sd": roc_sd,
        "pr_mean":  pr_m,  "pr_sd":  pr_sd,
    }

# ============================================================
# BLOCO 4 — MLP (Keras) PARA DADOS TABULARES (DII)
# ============================================================
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras import layers, models, callbacks, optimizers

def _compute_class_weights(y):
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.astype(int))
    return {0: float(weights[0]), 1: float(weights[1])}

def preparar_para_mlp(X_tr, X_val, X_te, meta):
    """
    Constrói matrizes para MLP:
      - mantém NUMÉRICAS já padronizadas
      - aplica ONE-HOT nas CATEGÓRICAS
    Retorna X_tr_mlp, X_val_mlp, X_te_mlp e o objeto ohe (para inspecionar depois).
    """
    num_idx = meta.get("numeric_cols_rel", [])
    cat_idx = meta.get("categorical_cols_rel", [])

    # separa
    Xtr_num = X_tr.iloc[:, num_idx].values
    Xval_num = X_val.iloc[:, num_idx].values
    Xte_num = X_te.iloc[:, num_idx].values

    Xtr_cat = X_tr.iloc[:, cat_idx].astype("int")
    Xval_cat = X_val.iloc[:, cat_idx].astype("int")
    Xte_cat  = X_te.iloc[:, cat_idx].astype("int")

    # One-hot só com treino para evitar vazamento
    # (se sua versão do sklearn for <1.2, troque sparse_output=False por sparse=False)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(Xtr_cat)

    Xtr_cat_oh = ohe.transform(Xtr_cat)
    Xval_cat_oh = ohe.transform(Xval_cat)
    Xte_cat_oh  = ohe.transform(Xte_cat)

    # concatena numérico + categórico
    X_tr_mlp  = np.hstack([Xtr_num,  Xtr_cat_oh])
    X_val_mlp = np.hstack([Xval_num, Xval_cat_oh])
    X_te_mlp  = np.hstack([Xte_num,  Xte_cat_oh])

    return X_tr_mlp, X_val_mlp, X_te_mlp, ohe

def construir_mlp_tabular(input_dim: int) -> tf.keras.Model:
    """MLP reforçado: GELU + BN + Dropout + L2 + AdamW + label smoothing."""
    reg = tf.keras.regularizers.l2(1e-5)
    init = "he_normal"

    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation=tf.nn.gelu, kernel_regularizer=reg, kernel_initializer=init)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, activation=tf.nn.gelu, kernel_regularizer=reg, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(32, activation=tf.nn.gelu, kernel_regularizer=reg, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    opt = optimizers.AdamW(learning_rate=3e-3, weight_decay=1e-4)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

def treinar_mlp_tabular(X_tr, y_tr, X_val, y_val, epochs=400, batch_size=32):
    """Treino com class weights, EarlyStopping, ReduceLROnPlateau."""
    model = construir_mlp_tabular(X_tr.shape[1])
    cw = _compute_class_weights(y_tr.values if isinstance(y_tr, pd.Series) else y_tr)

    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=35, restore_best_weights=True)
    cb_rlr   = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                                           min_lr=1e-5, verbose=1)

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=cw,
        callbacks=[cb_early, cb_rlr]
    )
    return model, hist

def _best_threshold_by_f1(y_true_val, scores_val):
    """Escolhe o limiar que maximiza F1 na validação (varre 0.20–0.80)."""
    thr_grid = np.linspace(0.20, 0.80, 121)
    best_t, best_f1 = 0.5, -1
    for t in thr_grid:
        f1 = f1_score(y_true_val, (scores_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def avaliar_mlp(model, X_val, y_val, X_te, y_te):
    """Ajusta limiar pela validação e reporta métricas no teste."""
    val_scores = model.predict(X_val).ravel()
    t_star, f1_val = _best_threshold_by_f1(y_val, val_scores)
    print(f"\n🔧 Limiar ótimo (val) para F1: t* = {t_star:.3f}  |  F1(val) = {f1_val:.3f}")

    y_score = model.predict(X_te).ravel()
    y_pred  = (y_score >= t_star).astype(int)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_te, y_score)
    pr  = average_precision_score(y_te, y_score)

    print("\n=== MLP — Avaliação no conjunto de teste (com limiar ajustado) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-macro: {f1:.4f}")
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr:.4f}")
    print("\nRelatório por classe:")
    print(classification_report(y_te, y_pred, digits=3, zero_division=0))

    return {"acc": acc, "f1": f1, "roc": roc, "pr": pr, "thr": t_star}, y_pred, y_score

def plotar_curvas_treino(hist):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history["loss"], label="Treino")
    plt.plot(hist.history["val_loss"], label="Validação")
    plt.title("Perda (Binary Crossentropy)")
    plt.xlabel("Épocas"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history["accuracy"], label="Treino")
    plt.plot(hist.history["val_accuracy"], label="Validação")
    plt.title("Acurácia")
    plt.xlabel("Épocas"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()






# ============================================================
# BLOCO – VISUALIZAÇÕES
# ============================================================
import seaborn as sns

def plot_confusao_binaria(y_true, y_pred, titulo="Matriz de Confusão (Binária)"):
    labels_txt = ["Saudável", "Arritmia"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels_txt, yticklabels=labels_txt, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(titulo + " — absolutos (anot.) e normalizados (cor)")
    plt.tight_layout()
    plt.show()

def plot_prob_heatmaps(proba_mat, y_true, title_prefix="Modelo × 100 execuções",
                       sort_by_mean=True):
    """
    Dois heatmaps: topo = Arritmia (y=1); base = Saudável (y=0).
    Cada coluna = uma execução (0..99); valores = P(arrítmico).
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list(
        "az_laranja", ["#85deff", "#f7f7f7", "#ffa676"], N=256
    )

    idx_arr = np.where(y_true == 1)[0]
    idx_sau = np.where(y_true == 0)[0]

    if sort_by_mean:
        mean_arr = proba_mat[idx_arr].mean(axis=1)
        mean_sau = proba_mat[idx_sau].mean(axis=1)
        idx_arr = idx_arr[np.argsort(-mean_arr)]
        idx_sau = idx_sau[np.argsort(mean_sau)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    sns.heatmap(proba_mat[idx_arr, :], vmin=0, vmax=1, cmap=cmap_custom,
                ax=axes[0], cbar_kws={"label": "Probabilidade de Arritmia"})
    axes[0].set_title("Arritmia", pad=8)
    axes[0].set_ylabel("Identificador do Paciente")

    sns.heatmap(proba_mat[idx_sau, :], vmin=0, vmax=1, cmap=cmap_custom,
                ax=axes[1], cbar_kws={"label": "Probabilidade de Arritmia"})
    axes[1].set_title("Saudável", pad=8)
    axes[1].set_ylabel("Identificador do Paciente")
    axes[1].set_xlabel("Número do Modelo")

    plt.suptitle(f"{title_prefix} — Probabilidade de Arritmia por Execução", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_distribuicao_classes(y_original, y_binario):
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    y_original.value_counts().sort_index().plot(kind='bar', ax=axs[0], color='#6fa8dc')
    axs[0].set_title("Distribuição Original das Classes")
    axs[0].set_xlabel("Classe (Multiclasse)")
    axs[0].set_ylabel("Número de Amostras")

    y_binario.value_counts().sort_index().plot(kind='bar', ax=axs[1], color='#93c47d')
    axs[1].set_title("Após Conversão Binária (0 = Saudável, 1 = Arritmia)")
    axs[1].set_xlabel("Classe (Binária)")
    axs[1].set_ylabel("Número de Amostras")

    plt.suptitle("Distribuição de Classes no Dataset Arrhythmia (DII)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_correlacao(X_pre):
    num = X_pre.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        print("Aviso: não há colunas numéricas para correlacionar após o pré-processamento.")
        return
    plt.figure(figsize=(10,8))
    corr = num.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlação entre variáveis (após seleção e pré-processamento)")
    plt.tight_layout()
    plt.show()

def plot_antes_depois_classificacao(y_multi_all: pd.Series,
                                    y_bin_all: pd.Series,
                                    titulo="Classificações — ANTES (multiclasse) × DEPOIS (binário) — GLOBAL"):
    counts_multi = (pd.Series(y_multi_all)
                      .astype(int)
                      .value_counts()
                      .sort_index())

    counts_bin = (pd.Series(y_bin_all)
                    .astype(int)
                    .value_counts()
                    .reindex([0, 1])
                    .fillna(0)
                    .astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    axes[0].bar(counts_multi.index.astype(str), counts_multi.values, color="#6fa8dc")
    axes[0].set_title("Distribuição ANTES (multiclasse — GLOBAL)")
    axes[0].set_xlabel("Classe (1..16)")
    axes[0].set_ylabel("Número de Amostras")

    axes[1].bar([0, 1], counts_bin.values, color="#93c47d")
    axes[1].set_title("DEPOIS (binarizado — GLOBAL)")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Saudável (0)", "Arritmia (1)"])
    axes[1].set_xlabel("Classe binária")

    plt.suptitle(titulo, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def plot_correlacao_dupla(X_raw: pd.DataFrame, X_pre: pd.DataFrame, title="Correlação — antes × depois"):
    num_raw = X_raw.select_dtypes(include=[np.number])
    num_pre = X_pre.select_dtypes(include=[np.number])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(num_raw.corr(), cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Antes (numéricas)")
    sns.heatmap(num_pre.corr(), cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Depois (numéricas)")
    plt.suptitle(title, y=0.95)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.show()

## ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    print("\n🚀 Pipeline DII | Rodando SVM, RF, HGB (100×) + MLP (tabular) + ENSEMBLES (4 modelos)\n")

    # ---------- Dados ----------
    df = carregar_dataset()
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas\n")

    X, y, meta = selecionar_colunas_DII(df)
    X_pre, y_pre, artef = preprocessar_dataset(X, y, meta)
    y_bin = to_binary_labels(y_pre, healthy_class=HEALTHY_CLASS)
    X_tr, X_val, X_te, y_tr, y_val, y_te = dividir_conjuntos(X_pre, y_bin)
    print(f"Treino: {X_tr.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_te.shape[0]}")

    # ---------- SVM / RF / HGB – 100x (VAL e TEST) ----------
    print("\n🧪 SVM — gerando probabilidades em 100 execuções (VAL e TEST)...")
    svm_proba_mat_val = svm_multi_runs_probs(X_tr, y_tr, X_val, n_runs=100)
    svm_proba_mat_te  = svm_multi_runs_probs(X_tr, y_tr, X_te,  n_runs=100)

    print("\n🧪 RF — gerando probabilidades em 100 execuções (VAL e TEST)...")
    rf_proba_mat_val = rf_multi_runs_probs(X_tr, y_tr, X_val, n_runs=100)
    rf_proba_mat_te  = rf_multi_runs_probs(X_tr, y_tr, X_te,  n_runs=100)

    print("\n🧪 HGB — gerando probabilidades em 100 execuções (VAL e TEST)...")
    hgb_proba_mat_val = hgb_multi_runs_probs(X_tr, y_tr, X_val, n_runs=100)
    hgb_proba_mat_te  = hgb_multi_runs_probs(X_tr, y_tr, X_te,  n_runs=100)

    # métricas médias ± desvio (100 execuções) — TEST
    svm_stats = avaliar_100_execucoes(y_te, svm_proba_mat_te, "SVM (100×)")
    rf_stats  = avaliar_100_execucoes(y_te, rf_proba_mat_te,  "Random Forest (100×)")
    hgb_stats = avaliar_100_execucoes(y_te, hgb_proba_mat_te, "HGB (100×)")

    # === "Ensemble interno" de cada modelo (média das 100 execuções) — TEST ===
    svm_ens_metrics, svm_pred_ens, svm_score_te = avaliar_ensemble_from_probas(
        y_te, svm_proba_mat_te, "SVM — ensemble (média 100×)"
    )
    rf_ens_metrics,  rf_pred_ens,  rf_score_te  = avaliar_ensemble_from_probas(
        y_te, rf_proba_mat_te,  "Random Forest — ensemble (média 100×)"
    )
    hgb_ens_metrics, hgb_pred_ens, hgb_score_te = avaliar_ensemble_from_probas(
        y_te, hgb_proba_mat_te, "HGB — ensemble (média 100×)"
    )

    # ---------- MLP tabular ----------
    print("\n🧠 Treinando MLP (Keras) sobre os dados tabulares DII (one-hot nas categóricas)...\n")
    X_tr_mlp, X_val_mlp, X_te_mlp, ohe = preparar_para_mlp(X_tr, X_val, X_te, meta)
    mlp_model, mlp_hist = treinar_mlp_tabular(X_tr_mlp, y_tr, X_val_mlp, y_val)

    # Probabilidades MLP em VAL/TEST (para os ensembles)
    mlp_score_val = mlp_model.predict(X_val_mlp).ravel()
    mlp_score_te  = mlp_model.predict(X_te_mlp).ravel()

    # Avaliação oficial da MLP (com threshold ótimo pela validação)
    mlp_metrics, mlp_pred, mlp_score_eval = avaliar_mlp(mlp_model, X_val_mlp, y_val, X_te_mlp, y_te)

    # ---------- Tabela comparativa (100× dos três clássicos) ----------
    def _fmt(mu, sd): return f"{mu:.3f} ± {sd:.3f}"
    tabela = pd.DataFrame([
        {
            "Modelo": "SVM (100×)",
            "Accuracy": _fmt(svm_stats["acc_mean"], svm_stats["acc_sd"]),
            "F1-macro": _fmt(svm_stats["f1m_mean"], svm_stats["f1m_sd"]),
            "ROC-AUC":  _fmt(svm_stats["roc_mean"], svm_stats["roc_sd"]) if not np.isnan(svm_stats["roc_mean"]) else "—",
            "PR-AUC":   _fmt(svm_stats["pr_mean"],  svm_stats["pr_sd"])  if not np.isnan(svm_stats["pr_mean"])  else "—",
        },
        {
            "Modelo": "Random Forest (100×)",
            "Accuracy": _fmt(rf_stats["acc_mean"], rf_stats["acc_sd"]),
            "F1-macro": _fmt(rf_stats["f1m_mean"], rf_stats["f1m_sd"]),
            "ROC-AUC":  _fmt(rf_stats["roc_mean"], rf_stats["roc_sd"]) if not np.isnan(rf_stats["roc_mean"]) else "—",
            "PR-AUC":   _fmt(rf_stats["pr_mean"],  rf_stats["pr_sd"])  if not np.isnan(rf_stats["pr_mean"])  else "—",
        },
        {
            "Modelo": "HGB (100×)",
            "Accuracy": _fmt(hgb_stats["acc_mean"], hgb_stats["acc_sd"]),
            "F1-macro": _fmt(hgb_stats["f1m_mean"], hgb_stats["f1m_sd"]),
            "ROC-AUC":  _fmt(hgb_stats["roc_mean"], hgb_stats["roc_sd"]) if not np.isnan(hgb_stats["roc_mean"]) else "—",
            "PR-AUC":   _fmt(hgb_stats["pr_mean"],  hgb_stats["pr_sd"])  if not np.isnan(hgb_stats["pr_mean"])  else "—",
        }
    ]).set_index("Modelo")
    print("\n================= MÉTRICAS MÉDIAS (100×) =================")
    print(tabela.to_string())

    # ========================================================
    # ENSEMBLES DOS 4 MODELOS (SVM + RF + HGB + MLP)
    #   - Soft Voting por média ponderada das probabilidades
    #   - Stacking com meta-modelo (LogisticRegression)
    # ========================================================

    # --- 1) Soft Voting (média ponderada) ---
    print("\n🗳️  Ensemble Soft (média ponderada das probabilidades dos 4 modelos)")

    # probabilidades médias por amostra
    svm_score_val = svm_proba_mat_val.mean(axis=1)
    rf_score_val  = rf_proba_mat_val.mean(axis=1)
    hgb_score_val = hgb_proba_mat_val.mean(axis=1)

    # pesos (ajuste fino opcional)
    w_svm, w_rf, w_hgb, w_mlp = 1.0, 1.0, 1.1, 0.9

    # mistura em VAL (para achar threshold ótimo por F1)
    ens_soft_val = (w_svm*svm_score_val + w_rf*rf_score_val + w_hgb*hgb_score_val + w_mlp*mlp_score_val) / (w_svm+w_rf+w_hgb+w_mlp)
    t_star_soft, f1_val_soft = _best_threshold_by_f1(y_val, ens_soft_val)
    print(f"🔧 Soft Voting — limiar ótimo (val): t*={t_star_soft:.3f} | F1(val)={f1_val_soft:.3f}")

    # aplica no TEST
    svm_score_te = svm_proba_mat_te.mean(axis=1)
    rf_score_te  = rf_proba_mat_te.mean(axis=1)
    hgb_score_te = hgb_proba_mat_te.mean(axis=1)
    ens_soft_te  = (w_svm*svm_score_te + w_rf*rf_score_te + w_hgb*hgb_score_te + w_mlp*mlp_score_te) / (w_svm+w_rf+w_hgb+w_mlp)
    y_pred_soft  = (ens_soft_te >= t_star_soft).astype(int)

    print("\n=== ENSEMBLE SOFT — Avaliação no TEST ===")
    _ = avaliar(y_te, y_pred_soft, titulo="Ensemble Soft (TEST)", y_score=ens_soft_te)

    # --- 2) Stacking (meta-modelo logístico) ---
    print("\n🧮  Ensemble Stacking (Logistic Regression nas probabilidades dos 4 modelos)")

    # Matriz de características para o meta-modelo (VAL e TEST)
    X_stack_val = np.vstack([
        svm_score_val,
        rf_score_val,
        hgb_score_val,
        mlp_score_val
    ]).T  # shape (n_val, 4)

    X_stack_te = np.vstack([
        svm_score_te,
        rf_score_te,
        hgb_score_te,
        mlp_score_te
    ]).T  # shape (n_test, 4)

    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    meta_clf.fit(X_stack_val, y_val)

    stack_score_te = meta_clf.predict_proba(X_stack_te)[:, 1]
    t_star_stack, f1_val_stack = _best_threshold_by_f1(y_val, meta_clf.predict_proba(X_stack_val)[:,1])
    print(f"🔧 Stacking — limiar ótimo (val): t*={t_star_stack:.3f} | F1(val)={f1_val_stack:.3f}")

    y_pred_stack = (stack_score_te >= t_star_stack).astype(int)
    print("\n=== ENSEMBLE STACKING — Avaliação no TEST ===")
    _ = avaliar(y_te, y_pred_stack, titulo="Ensemble Stacking (TEST)", y_score=stack_score_te)

    # ---------- Plots globais e diagnósticos ----------
    y_bin_all = to_binary_labels(y, healthy_class=HEALTHY_CLASS)
    plot_antes_depois_classificacao(
        y_multi_all=y.astype(int),
        y_bin_all=y_bin_all.astype(int),
        titulo="Antes (multiclasse) × Depois (binário) — GLOBAL (dataset inteiro)"
    )
    plot_correlacao(X_pre)

    print("\n🎨 Gerando heatmaps (100 execuções)...")
    plot_prob_heatmaps(svm_proba_mat_te, y_te, title_prefix="SVM × 100 execuções")
    plot_prob_heatmaps(rf_proba_mat_te,  y_te, title_prefix="Random Forest × 100 execuções")
    plot_prob_heatmaps(hgb_proba_mat_te, y_te, title_prefix="HGB × 100 execuções")


# ============================================================
# SALVAR IA PARA USO NO SITE (PERSISTÊNCIA DE ARTEFATOS)
# ============================================================
from pathlib import Path
from joblib import dump
import json

MODELS_DIR = Path("./models_site")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("\n💾 Salvando artefatos da IA para uso no site...")

# 1) Salvar ordem das features (como chegam do seu extrator/CSV)
feature_order = list(X.columns)  # ex.: ['f5','f6',...]
with open(MODELS_DIR / "feature_order.json", "w", encoding="utf-8") as f:
    json.dump(feature_order, f, ensure_ascii=False, indent=2)

# 2) Salvar pré-processamento (imputers + scaler) e metadados
preproc_bundle = {
    "num_imputer": artef["num_imputer"],
    "cat_imputer": artef["cat_imputer"],
    "scaler": artef["scaler"],
    "meta": meta,  # índices num/cat/amp e flags que você já usa
}
dump(preproc_bundle, MODELS_DIR / "preproc.joblib")


# 3) Treinar UMA cópia de cada modelo clássico (SVM/RF/HGB) para inferência online
#    (assim não precisamos rodar 100× no site)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

X_base = pd.concat([X_tr, X_val], axis=0)
y_base = pd.concat([y_tr, y_val], axis=0)

svm_frozen = SVC(C=2.0, kernel="rbf", gamma="scale",
                 class_weight="balanced", probability=True, random_state=SEED)
svm_frozen.fit(X_base, y_base)
dump(svm_frozen, MODELS_DIR / "svm.joblib")

rf_frozen = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", n_jobs=-1, random_state=SEED
)
rf_frozen.fit(X_base, y_base)
dump(rf_frozen, MODELS_DIR / "rf.joblib")

hgb_frozen = HistGradientBoostingClassifier(
    learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20,
    l2_regularization=1e-3, early_stopping=True, validation_fraction=0.15, random_state=SEED
)
sw_base = compute_sample_weight(class_weight="balanced", y=y_base)
hgb_frozen.fit(X_base, y_base, sample_weight=sw_base)
dump(hgb_frozen, MODELS_DIR / "hgb.joblib")

# 4) Salvar MLP Keras e o OneHotEncoder usado por ela
mlp_model.save(MODELS_DIR / "mlp.h5")
dump(ohe, MODELS_DIR / "mlp_ohe.joblib")  # o mesmo OHE que você ajustou no treino

# 5) Salvar os DOIS ensembles que você ajustou:
#    (a) Soft Voting: pesos e limiar ótimo (t*)
soft_cfg = {
    "weights": [float(w) for w in [w_svm, w_rf, w_hgb, w_mlp]],   # definidos acima
    "threshold": float(t_star_soft)
}
with open(MODELS_DIR / "soft_config.json", "w", encoding="utf-8") as f:
    json.dump(soft_cfg, f, ensure_ascii=False, indent=2)

#    (b) Stacking: meta-classificador LogReg treinado na validação + limiar ótimo (t*)
dump(meta_clf, MODELS_DIR / "stack_meta.joblib")
with open(MODELS_DIR / "stack_config.json", "w", encoding="utf-8") as f:
    json.dump({"threshold": float(t_star_stack)}, f, ensure_ascii=False, indent=2)

print("✅ Artefatos salvos em ./models_site")

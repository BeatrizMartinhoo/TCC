# arrhythmia_infer.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

# Procura os artefatos em ./models_site 
MODELS_DIR = Path(__file__).resolve().parent / "models_site"

# Variáveis globais (lazy-load)
_feature_order = None
_preproc = None
_svm = None
_rf = None
_hgb = None
_mlp = None
_ohe = None
_soft_cfg = None
_stack_meta = None
_stack_cfg = None
_meta = None
_num_imp = None
_cat_imp = None
_scaler = None

def _ensure_loaded():
    """
    Carrega os artefatos somente quando necessário, com mensagens claras caso
    algum arquivo esteja ausente ou com estrutura diferente do treino.
    """
    global _feature_order, _preproc, _svm, _rf, _hgb, _mlp, _ohe
    global _soft_cfg, _stack_meta, _stack_cfg
    global _meta, _num_imp, _cat_imp, _scaler

    if _feature_order is not None:
        return  # já carregado

    missing = []
    def _need(p): 
        if not (MODELS_DIR / p).exists(): missing.append(p)

    # lista de arquivos esperados
    expected = [
        "feature_order.json",
        "preproc.joblib",
        "svm.joblib", "rf.joblib", "hgb.joblib",
        "mlp.h5", "mlp_ohe.joblib",
        "soft_config.json",
        "stack_meta.joblib", "stack_config.json",
    ]
    for p in expected: _need(p)
    if missing:
        raise FileNotFoundError(
            "Artefatos de IA não encontrados. Coloque estes arquivos em ./models_site:\n  - " +
            "\n  - ".join(missing)
        )

    # Carregamentos
    _feature_order = json.loads((MODELS_DIR / "feature_order.json").read_text(encoding="utf-8"))

    from joblib import load as _jload
    _preproc = _jload(MODELS_DIR / "preproc.joblib")

    # modelos clássicos
    _svm = _jload(MODELS_DIR / "svm.joblib")
    _rf  = _jload(MODELS_DIR / "rf.joblib")
    _hgb = _jload(MODELS_DIR / "hgb.joblib")

    # MLP + OHE
    from tensorflow.keras.models import load_model as _kload
    _mlp = _kload(MODELS_DIR / "mlp.h5")
    _ohe = _jload(MODELS_DIR / "mlp_ohe.joblib")

    _soft_cfg  = json.loads((MODELS_DIR / "soft_config.json").read_text(encoding="utf-8"))
    _stack_meta = _jload(MODELS_DIR / "stack_meta.joblib")
    _stack_cfg  = json.loads((MODELS_DIR / "stack_config.json").read_text(encoding="utf-8"))

    # ===== meta e pré-processamento (com tolerância) =====
    _meta = _preproc.get("meta", {})
    _num_imp = _preproc.get("num_imputer", None)
    _cat_imp = _preproc.get("cat_imputer", None)
    _scaler  = _preproc.get("scaler", None)

    # fallback seguro, caso algum item não esteja no joblib por ser antigo
    from sklearn.impute import SimpleImputer
    if _num_imp is None:
        _num_imp = SimpleImputer(strategy="median")
    if _cat_imp is None:
        _cat_imp = SimpleImputer(strategy="most_frequent")
    # _scaler pode ser None (nesse caso não padroniza)




def _preprocess_one(sample_dict: dict) -> pd.Series:
    _ensure_loaded()

    # 🔍 garante que só usamos colunas esperadas (mesmas do treino)
    sample_filtered = {k: sample_dict.get(k, np.nan) for k in _feature_order}
    X = pd.DataFrame([sample_filtered], columns=_feature_order)
    X = X.fillna(0)  # evita NaN se algo não for medido




    # 2) aplicar mesma lógica do preprocessar_dataset (imputers + scaler)
    cat_idx = _meta.get("categorical_cols_rel", [])
    num_idx = _meta.get("numeric_cols_rel", [])

    # fallback: se meta não tiver índices, trata tudo como numérico
    if not num_idx and not cat_idx:
        num_idx = list(range(X.shape[1]))
        cat_idx = []

    X_num = X.iloc[:, num_idx].apply(pd.to_numeric, errors="coerce") if num_idx else pd.DataFrame(index=X.index)
    X_cat = X.iloc[:, cat_idx].copy() if cat_idx else pd.DataFrame(index=X.index)
    for c in X_cat.columns:
        X_cat[c] = pd.to_numeric(X_cat[c], errors="coerce")

    # fit de segurança (artefatos antigos podem vir “desfitados”)
    if X_num.shape[1] > 0 and not hasattr(_num_imp, "statistics_"):
        _num_imp.fit(X_num)
    if X_cat.shape[1] > 0 and not hasattr(_cat_imp, "statistics_"):
        _cat_imp.fit(X_cat)

    X_num_imp = (pd.DataFrame(_num_imp.transform(X_num), columns=X_num.columns, index=X.index)
                 if X_num.shape[1] > 0 else X_num)
    X_cat_imp = (pd.DataFrame(_cat_imp.transform(X_cat), columns=X_cat.columns, index=X.index)
                 if X_cat.shape[1] > 0 else X_cat)
    if X_cat_imp.shape[1] > 0:
        X_cat_imp = X_cat_imp.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0).astype(int)

    if X_num_imp.shape[1] > 0 and _scaler is not None:
        if not hasattr(_scaler, "scale_"):
            _scaler.fit(X_num_imp)
        X_num_std = pd.DataFrame(_scaler.transform(X_num_imp), columns=X_num_imp.columns, index=X.index)
    else:
        X_num_std = X_num_imp

    # reconstruir na ordem original (num padronizado + cat imputado)
    partes = []
    for j in range(X.shape[1]):
        colname = X.columns[j]
        if j in num_idx:
            partes.append(X_num_std[[colname]])
        else:
            partes.append(X_cat_imp[[colname]] if colname in X_cat_imp.columns else pd.DataFrame({colname: [0]}, index=X.index))
    X_pre = pd.concat(partes, axis=1)

    # 🔧 Garanta que os nomes das colunas batem com os do treino
    try:
        expected_names = _meta.get("all_features", _feature_order)
        if isinstance(expected_names, list) and len(expected_names) == X_pre.shape[1]:
            X_pre.columns = expected_names
        else:
            # fallback seguro: use a ordem de treino (f5, f6, …)
            X_pre.columns = _feature_order
    except Exception:
        X_pre.columns = _feature_order

    return X_pre.iloc[0]


def _mlp_matrix_from_pre(X_pre_row: pd.Series):
    """Constrói vetor para MLP (num + one-hot cat), como no treino."""
    _ensure_loaded()
    num_idx = _meta.get("numeric_cols_rel", [])
    cat_idx = _meta.get("categorical_cols_rel", [])

    X_num = np.asarray(X_pre_row.iloc[num_idx], dtype=float).reshape(1, -1)

    # nomes categóricos exatamente como no treino
    cat_names = [_feature_order[i] for i in cat_idx]
    X_cat = pd.DataFrame(
        [X_pre_row.iloc[cat_idx].astype(int).values],
        columns=cat_names
    )

    # reindexa para a ordem/nome esperados pelo OHE treinado (quando disponível)
    if hasattr(_ohe, "feature_names_in_"):
        X_cat = X_cat.reindex(columns=_ohe.feature_names_in_, fill_value=0)

    X_cat_oh = _ohe.transform(X_cat)
    return np.hstack([X_num, X_cat_oh])


def _scores_from_bases(X_pre_row: pd.Series):
    """Probabilidades dos 4 modelos base (classe positiva = 1)."""
    _ensure_loaded()

    # scikit-learn (SVM, RF, HGB) — com nomes de colunas
    x_scikit = X_pre_row.to_frame().T  # DataFrame 1xN preservando .index como nomes
    p_svm = float(_svm.predict_proba(x_scikit)[:, 1][0])
    p_rf  = float(_rf.predict_proba(x_scikit)[:, 1][0])
    p_hgb = float(_hgb.predict_proba(x_scikit)[:, 1][0])

    # Keras MLP
    x_mlp = _mlp_matrix_from_pre(X_pre_row)
    p_mlp = float(_mlp.predict(x_mlp, verbose=0).ravel()[0])

    return p_svm, p_rf, p_hgb, p_mlp


def predict_from_features(sample: dict, method: str = "stacking", thr_override: float | None = None):
    _ensure_loaded()

    # 1) pré-processa a amostra como no treino
    X_pre_row = _preprocess_one(sample)

    # 2) probabilidades dos modelos base
    p_svm, p_rf, p_hgb, p_mlp = _scores_from_bases(X_pre_row)

    # 3) ensemble
    if method == "soft":
        w = _soft_cfg.get("weights", [0.25, 0.25, 0.25, 0.25])
        p_final = float(w[0]*p_svm + w[1]*p_rf + w[2]*p_hgb + w[3]*p_mlp)
    else:  # stacking (meta-modelo)
        x_meta = np.array([[p_svm, p_rf, p_hgb, p_mlp]], dtype=float)
        p_final = float(_stack_meta.predict_proba(x_meta)[0, 1])

    # 4) limiar (mais conservador por padrão)
    # limiar padrão mais conservador (75%)
    thr_cfg = float(_stack_cfg.get("threshold", 0.75))
    thr_base = float(thr_override) if (thr_override is not None) else thr_cfg
    # protege contra valores extremos por engano
    thr = max(0.50, min(thr_base, 0.95))


    # 5) --- VETO FISIOLÓGICO (anti-falso-positivo, quando prob. não é muito alta) ---
    # usa as features clínicas se disponíveis (em ms/bpm)
    qrs = sample.get("f5")  or sample.get("qrs_duration_ms")
    pr  = sample.get("f6")  or sample.get("pr_interval_ms")
    qt  = sample.get("f7")  or sample.get("qt_interval_ms")
    pd  = sample.get("f9")  or sample.get("p_duration_ms")
    hr  = sample.get("f15") or sample.get("heart_rate_bpm")

    # flags discretas (0/1) das “ragged/diphasic” quando existirem
    f34 = sample.get("f34", 0); f35 = sample.get("f35", 0)
    f36 = sample.get("f36", 0); f37 = sample.get("f37", 0); f38 = sample.get("f38", 0)

    physiologically_normal = True
    try:
        if qrs is not None: physiologically_normal &= (80 <= float(qrs) <= 120)
        if pr  is not None: physiologically_normal &= (120 <= float(pr) <= 200)
        if qt  is not None: physiologically_normal &= (340 <= float(qt) <= 440)
        if pd  is not None: physiologically_normal &= (80  <= float(pd) <= 130)
        if hr  is not None: physiologically_normal &= (55  <= float(hr) <= 100)
        # se nenhum sinal de “irregularidade morfológica”
        irregular_flags = sum(int(bool(x)) for x in [f34, f35, f36, f37, f38])
        physiologically_normal &= (irregular_flags == 0)
    except Exception:
        physiologically_normal = False  # se algo deu ruim, não aplica o veto

    # Se o sinal é fisiologicamente normal e a confiança NÃO é alta, classifique como normal
    if physiologically_normal and p_final < 0.85:
        label = 0
    else:
        label = int(p_final >= thr)

    detail = {
        "p_svm": p_svm, "p_rf": p_rf, "p_hgb": p_hgb, "p_mlp": p_mlp,
        "ensemble_prob": p_final, "threshold_used": thr,
        "physio_normal_gate": bool(physiologically_normal)
    }
    return {"label": label, "score": p_final, "threshold": thr, "detail": detail}

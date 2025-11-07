# ecg_features.py
# -----------------------------------------------------------------------------
# 1) Parse RAW->DF no formato do notebook: t_ms,value,lo_plus,lo_minus
# 2) Filtros "como no notebook": LO-mask -> detrend -> notch(60Hz) -> bandpass(0.7-40Hz) -> SavGol
# 3) Leitura de CSV no formato do notebook
# 4) Extração de features (pode usar coluna filtrada 'ecg_proc' gerada aqui)
#
# Observações:
# - Os filtros aqui replicam a pipeline típica do notebook de ECG:
#   • Remoção de trechos com LO± (interpolação)
#   • detrend linear (remove offset)
#   • notch 60 Hz (Q padrão 30)
#   • band-pass Butterworth 0.7–40 Hz (ordem 3)
#   • suavização Savitzky-Golay (~0.12 s, polyorder=3)
# - Se quiser usar o NeuroKit2, mantenha 'use_neurokit=True' e passe a coluna filtrada.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd

# SciPy para filtros
try:
    from scipy.signal import butter, filtfilt, iirnotch, savgol_filter, detrend as sp_detrend
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# NeuroKit2 é opcional — usado se ativado na extração
try:
    import neurokit2 as nk  # type: ignore
    _NK_OK = True
except Exception:
    _NK_OK = False

def _nanmad(a, axis=None):
    a = np.asarray(a, dtype=float)
    med = np.nanmedian(a, axis=axis)
    return np.nanmedian(np.abs(a - med), axis=axis)

def _robust_ms(series_ms: np.ndarray, lo: float, hi: float) -> float:
    """Mede por mediana, remove outliers por MAD e faz clipping fisiológico suave."""
    x = np.asarray(series_ms, dtype=float)
    m = np.nanmedian(x)
    mad = _nanmad(x)
    if np.isfinite(mad) and mad > 0:
        mask = np.abs(x - m) <= 3.5*1.4826*mad
        x = x[mask]
    m = np.nanmedian(x) if x.size else np.nan
    if np.isfinite(m):
        m = float(np.clip(m, lo, hi))
    return m


# =========================
# 1) PARSERS DE LINHAS RAW
# =========================

def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x.strip())
    except Exception:
        return None


def parse_raw_line(line: str) -> pd.DataFrame:
    """
    Converte uma linha do tipo:
      RAW:19964,1398,0,0;19968,1366,0,0;19972,1344,0,0;...
    para um DataFrame com colunas:
      t_ms, value, lo_plus, lo_minus
    """
    if not line:
        return _empty_notebook_df()

    line = line.strip()
    if not line.startswith("RAW:"):
        return _empty_notebook_df()

    payload = line.split("RAW:", 1)[1]
    chunks = [c for c in payload.split(";") if c.strip()]

    rows: List[Tuple[int, int, int, int]] = []
    for c in chunks:
        parts = c.split(",")
        if len(parts) < 2:
            continue
        t_ms = _safe_int(parts[0])
        adc  = _safe_int(parts[1])
        lp   = _safe_int(parts[2]) if len(parts) > 2 else 0
        lm   = _safe_int(parts[3]) if len(parts) > 3 else 0
        if t_ms is None or adc is None:
            continue
        rows.append((t_ms, adc, lp or 0, lm or 0))

    if not rows:
        return _empty_notebook_df()

    df = pd.DataFrame(rows, columns=["t_ms", "value", "lo_plus", "lo_minus"])
    df = (df.sort_values("t_ms")
            .drop_duplicates(subset=["t_ms"])
            .reset_index(drop=True))
    return df


def _empty_notebook_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["t_ms", "value", "lo_plus", "lo_minus"])


# ==================================================
# 2) ACUMULADOR (para ir juntando várias linhas RAW)
# ==================================================

@dataclass
class RawAccumulator:
    """
    Buffer de amostras já no formato do notebook.
    """
    df: pd.DataFrame = field(default_factory=_empty_notebook_df)

    def add_line(self, line: str) -> None:
        part = parse_raw_line(line)
        if part.empty:
            return
        if self.df.empty:
            self.df = part
        else:
            self.df = pd.concat([self.df, part], ignore_index=True)
            self.df = (self.df.sort_values("t_ms")
                               .drop_duplicates(subset=["t_ms"])
                               .reset_index(drop=True))

    def clear(self) -> None:
        self.df = _empty_notebook_df()

    def get_df(self) -> pd.DataFrame:
        return self.df.copy()

    def to_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)


# =========================================================
# 3) UTILITÁRIOS OPCIONAIS (não usados automaticamente)
# =========================================================

def adc_to_millivolts(
    adc_values: Union[pd.Series, np.ndarray, List[int]],
    bits: int = 12,
    vref: float = 3.3,
    gain: float = 1.0,
    remove_dc: bool = True
) -> np.ndarray:
    """
    Converte ADC cru para mV (não aplicado automaticamente).
    """
    x = np.asarray(adc_values, dtype=float)
    full_scale = (2 ** bits) - 1
    mv = (x / full_scale) * vref * 1000.0 / max(gain, 1e-9)
    if remove_dc:
        mv = mv - np.nanmedian(mv)
    return mv


def build_notebook_df_from_adc(
    t_ms: Union[pd.Series, np.ndarray, List[int]],
    adc: Union[pd.Series, np.ndarray, List[int]],
    lo_plus: Optional[Union[pd.Series, np.ndarray, List[int]]] = None,
    lo_minus: Optional[Union[pd.Series, np.ndarray, List[int]]] = None,
) -> pd.DataFrame:
    t = np.asarray(t_ms, dtype=int)
    v = np.asarray(adc, dtype=int)
    lp = np.zeros_like(v, dtype=int) if lo_plus is None else np.asarray(lo_plus, dtype=int)
    lm = np.zeros_like(v, dtype=int) if lo_minus is None else np.asarray(lo_minus, dtype=int)

    df = pd.DataFrame({
        "t_ms": t,
        "value": v,
        "lo_plus": lp,
        "lo_minus": lm
    }).sort_values("t_ms").drop_duplicates(subset=["t_ms"]).reset_index(drop=True)
    return df


# ===============================================
# 4) FILTROS "COMO NO NOTEBOOK"
# ===============================================

def infer_fs_from_tms(t_ms_col: Union[pd.Series, np.ndarray, List[int], None]) -> Optional[float]:
    if t_ms_col is None:
        return None
    t = np.asarray(t_ms_col, dtype=float) / 1000.0
    if t.size < 2:
        return None
    dt = np.diff(t)
    med = np.nanmedian(dt) if dt.size else np.nan
    if not np.isfinite(med) or med <= 0:
        return None
    return 1.0 / med


def _odd_window(n: int, min_odd: int = 5) -> int:
    """Gera tamanho de janela ímpar >= min_odd."""
    n = max(min_odd, int(round(n)))
    if n % 2 == 0:
        n += 1
    return n


def _mask_lo_and_interpolate(x: np.ndarray, lo_plus: np.ndarray, lo_minus: np.ndarray) -> np.ndarray:
    """
    Marca trechos com LO± != 0 como NaN e interpola linearmente.
    Onde não houver como interpolar, preenche com mediana.
    """
    y = np.asarray(x, dtype=float).copy()
    bad = (np.asarray(lo_plus) != 0) | (np.asarray(lo_minus) != 0)
    if bad.any():
        y[bad] = np.nan
        # interpola apenas nos NaNs internos
        idx = np.arange(y.size)
        good = np.isfinite(y)
        if good.sum() >= 2:
            y = np.interp(idx, idx[good], y[good])
        # em caso de bordas longas sem good, garante preenchimento
        if np.isnan(y).any():
            y[np.isnan(y)] = np.nanmedian(y[np.isfinite(y)])
    return y


def _apply_detrend(x: np.ndarray) -> np.ndarray:
    if not _SCIPY_OK or len(x) < 8:
        return x - np.nanmedian(x)
    try:
        return sp_detrend(x, type="linear")
    except Exception:
        return x - np.nanmedian(x)


def _apply_notch(x: np.ndarray, fs: float, f0: float = 60.0, Q: float = 30.0) -> np.ndarray:
    if not _SCIPY_OK or len(x) < 16:
        return x
    try:
        b, a = iirnotch(w0=f0/(fs/2.0), Q=max(Q, 1e-6))
        return filtfilt(b, a, x, method="gust")
    except Exception:
        return x


def _apply_bandpass(x: np.ndarray, fs: float, low: float = 0.7, high: float = 40.0, order: int = 3) -> np.ndarray:
    if not _SCIPY_OK or len(x) < max(64, 10 * order):
        return x
    try:
        wn = [max(1e-6, low/(fs/2.0)), min(0.999, high/(fs/2.0))]
        b, a = butter(order, wn, btype="band")
        return filtfilt(b, a, x, method="gust")
    except Exception:
        return x


def _apply_savgol(x: np.ndarray, fs: float, win_sec: float = 0.12, polyorder: int = 3) -> np.ndarray:
    if not _SCIPY_OK or len(x) < 9:
        return x
    try:
        win = _odd_window(win_sec * fs, min_odd=5)
        if win >= len(x):  # garante janela válida
            win = _odd_window(len(x) - 1, min_odd=5)
        if win < 5:
            return x
        return savgol_filter(x, window_length=win, polyorder=min(polyorder, win-2), mode="interp")
    except Exception:
        return x


def filter_df_like_notebook(df: pd.DataFrame,
                            fs: float,
                            signal_col: str = "value",
                            out_col: str = "ecg_proc",
                            assume_is_adc: bool = True,
                            respect_lo_mask: bool = True,
                            notch_f0_hz: float = 60.0,
                            notch_Q: float = 30.0,
                            bp_low_hz: float = 0.67,
                            bp_high_hz: float = 30.0,
                            bp_order: int = 3,
                            sg_win_s: float = 0.12,
                            sg_poly: int = 3) -> pd.DataFrame:
    """
    Filtro mais conservador para ECG de única derivação (AD8232):
    - Notch 60 Hz (Q=30)
    - Bandpass 0.67–30 Hz (reduz superlargura de QRS)
    - Savitzky-Golay suave para reduzir jitter sem alargar picos
    """
    import numpy as np
    from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt, savgol_filter

    x = df[signal_col].astype(float).to_numpy()

    # Notch
    b, a = iirnotch(w0=notch_f0_hz/(fs/2), Q=notch_Q)
    x = filtfilt(b, a, x)

    # Bandpass
    sos = butter(bp_order, [bp_low_hz/(fs/2), bp_high_hz/(fs/2)], btype="band", output="sos")
    x = sosfiltfilt(sos, x)

    # SG
    win = max(5, int(round(sg_win_s*fs)) | 1)  # ímpar
    x = savgol_filter(x, window_length=win, polyorder=sg_poly)

    y = x.copy()
    out = df.copy()
    out[out_col] = y

    if respect_lo_mask and {"lo_plus","lo_minus"}.issubset(df.columns):
        bad = (df["lo_plus"]>0) | (df["lo_minus"]>0)
        out.loc[bad, out_col] = np.nan  # zera regiões ruins
    return out


# ===============================================
# 5) LEITURA DE CSV NO FORMATO DO NOTEBOOK
# ===============================================

def load_notebook_csv(path: str) -> pd.DataFrame:
    """
    Lê o CSV no formato do notebook e garante colunas/tipos/ordem.
    Esperado: t_ms,value,lo_plus,lo_minus
    """
    df = pd.read_csv(path)
    # Normaliza nomes
    cols_norm = {c.lower().strip(): c for c in df.columns}
    rename_map = {}
    for want in ["t_ms", "value", "lo_plus", "lo_minus"]:
        if want not in df.columns and want in cols_norm:
            rename_map[cols_norm[want]] = want
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ["t_ms", "value", "lo_plus", "lo_minus"]:
        if c not in df.columns:
            df[c] = 0

    df = (
        df.assign(
            t_ms=lambda d: d["t_ms"].astype(int, errors="ignore"),
            value=lambda d: d["value"].astype(float, errors="ignore"),
            lo_plus=lambda d: d["lo_plus"].fillna(0).astype(int, errors="ignore"),
            lo_minus=lambda d: d["lo_minus"].fillna(0).astype(int, errors="ignore"),
        )[["t_ms", "value", "lo_plus", "lo_minus"]]
        .drop_duplicates(subset=["t_ms"])
        .sort_values("t_ms")
        .reset_index(drop=True)
    )
    return df


# ======================================================
# 6) EXTRAÇÃO DE FEATURES (agora com filtros antes)
# ======================================================

def extract_features_from_df(
    df: pd.DataFrame,
    *,
    fs: Optional[float] = None,
    # coluna de entrada (ADC cru do notebook)
    signal_col: str = "value",
    # aplicar filtros do notebook aqui:
    apply_filters: bool = True,
    filtered_col_out: str = "ecg_proc",
    # conversão ADC -> mV antes de filtrar:
    assume_is_adc: bool = True,
    adc_bits: int = 12,
    vref: float = 3.3,
    gain: float = 1.0,
    remove_dc_after_adc: bool = True,
    # parâmetros dos filtros:
    notch_f0_hz: float = 60.0,
    notch_Q: float = 30.0,
    bp_low_hz: float = 0.7,
    bp_high_hz: float = 40.0,
    bp_order: int = 3,
    sg_win_s: float = 0.12,
    sg_poly: int = 3,
    respect_lo_mask: bool = True,
    # delineação com NeuroKit (usa a série filtrada):
    use_neurokit: bool = True,
) -> Dict[str, float]:
    """
    Extrai features clínicas a partir do DF no formato do notebook.
    Calcula f5..f9, f15, f28..f39 (médias por batimento), preservando NaN quando
    objetivamente não mensurável (NÃO substitui por 0.0).
    """
    if df is None or df.empty:
        return _nan_features("DF vazio")

    if fs is None:
        fs = infer_fs_from_tms(df.get("t_ms"))
    if fs is None or fs <= 1.0:
        return _nan_features("fs inválido")

    x_df = df.copy()
    if apply_filters:
        x_df = filter_df_like_notebook(
            x_df,
            fs=fs,
            signal_col=signal_col,
            out_col=filtered_col_out,
            assume_is_adc=assume_is_adc,
            adc_bits=adc_bits,
            vref=vref,
            gain=gain,
            remove_dc_after_adc=remove_dc_after_adc,
            apply_notch=True,
            notch_f0_hz=notch_f0_hz,
            notch_Q=notch_Q,
            bp_low_hz=bp_low_hz,
            bp_high_hz=bp_high_hz,
            bp_order=bp_order,
            sg_win_s=sg_win_s,
            sg_poly=sg_poly,
            respect_lo_mask=respect_lo_mask,
        )
        use_col = filtered_col_out
    else:
        use_col = signal_col

    if use_col not in x_df.columns:
        return _nan_features(f"coluna '{use_col}' ausente")

    x = x_df[use_col].to_numpy(dtype=float)

    # ---------- Inicializa saídas ----------
    f = {
        # globais
        "heart_rate_bpm": np.nan,    # f15
        "qrs_duration_ms": np.nan,   # f5
        "pr_interval_ms": np.nan,    # f6
        "qt_interval_ms": np.nan,    # f7
        "t_duration_ms": np.nan,     # f8
        "p_duration_ms": np.nan,     # f9
        "num_beats": np.nan,
        # DII específicos (larguras/flags)
        "q_width_ms": np.nan,        # f28
        "r_width_ms": np.nan,        # f29
        "s_width_ms": np.nan,        # f30
        "r2_width_ms": np.nan,       # f31 (R')
        "s2_width_ms": np.nan,       # f32 (S')
        "num_intrinsic_defl": np.nan,# f33
        "ragged_r": np.nan,          # f34
        "diphasic_r": np.nan,        # f35
        "ragged_p": np.nan,          # f36
        "diphasic_p": np.nan,        # f37
        "ragged_t": np.nan,          # f38
        "diphasic_t": np.nan,        # f39
        "fs_hz": float(fs),
    }

    rpeaks = None
    signals = None
    # ---------- Tenta usar NeuroKit2 ----------
    if use_neurokit and _NK_OK:
        try:
            signals, info = nk.ecg_process(x, sampling_rate=fs)
            rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            if "ECG_Rate" in signals:
                f["heart_rate_bpm"] = float(np.nanmean(signals["ECG_Rate"].values))

            # durações/intervalos via NK
            def _meddur(on_key, off_key):
                if on_key in signals.columns and off_key in signals.columns:
                    on_beg, on_end = _indices_from_binary_edges((signals[on_key].values > 0))
                    off_beg, off_end = _indices_from_binary_edges((signals[off_key].values > 0))
                    n = min(len(on_beg), len(off_beg))
                    if n > 0:
                        d = (off_end[:n] - on_beg[:n]) / fs * 1000.0
                        d = d[(d > 30) & (d < 250)]  # restrição fisiológica: 30–250 ms
                        return _safe_nanmean(d)
                return np.nan

            def _medint(a_key, b_key):
                if a_key in signals.columns and b_key in signals.columns:
                    a_beg, a_end = _indices_from_binary_edges((signals[a_key].values > 0))
                    b_beg, b_end = _indices_from_binary_edges((signals[b_key].values > 0))
                    n = min(len(a_end), len(b_beg))
                    if n > 0:
                        d = (b_beg[:n] - a_end[:n]) / fs * 1000.0
                        d = d[(d > 40) & (d < 500)]  # restrição fisiológica: 40–500 ms
                        return _safe_nanmean(d)
                return np.nan


            f["qrs_duration_ms"] = _meddur("ECG_QRS_Complex_Onsets", "ECG_QRS_Complex_Offsets")  # f5
            f["pr_interval_ms"]  = _medint("ECG_P_Offsets", "ECG_QRS_Complex_Onsets")             # f6
            f["qt_interval_ms"]  = _medint("ECG_QRS_Complex_Onsets", "ECG_T_Offsets")            # f7
            f["t_duration_ms"]   = _meddur("ECG_T_Onsets", "ECG_T_Offsets")                      # f8
            f["p_duration_ms"]   = _meddur("ECG_P_Onsets", "ECG_P_Offsets")                      # f9

            # ---------- DII específicos dentro do QRS ----------
            qrs_wins = _qrs_windows_from_signals(signals, fs)
            if qrs_wins is None and rpeaks is not None and rpeaks.size:
                qrs_wins = _simple_qrs_windows_from_rpeaks(rpeaks, fs, len(x))

            q_widths = []
            r_widths = []
            s_widths = []
            r2_widths = []
            s2_widths = []
            deflections = []
            ragR = []
            dipR = []

            if qrs_wins is not None and len(qrs_wins):
                for (a, b) in qrs_wins:
                    # 1) pico R principal
                    (imax, vmax), (imin, vmin) = _find_local_extrema(x, a, b)
                    if not np.isfinite(imax):
                        continue
                    rp = int(imax)

                    # 2) Q (mínimo antes de R) — se não achar, tenta mínimo local no lado esquerdo
                    _, (q_idx, q_val) = _find_local_extrema(x, a, rp)
                    if not np.isfinite(q_idx) and rp - a > 3:
                        try:
                            q_idx = a + int(np.nanargmin(x[a:rp]))
                            q_val = x[int(q_idx)]
                        except Exception:
                            q_idx = np.nan

                    # 3) S (mínimo após R) — se não achar, tenta mínimo local no lado direito
                    _, (s_idx, s_val) = _find_local_extrema(x, rp, b)
                    if not np.isfinite(s_idx) and b - rp > 3:
                        try:
                            s_idx = rp + int(np.nanargmin(x[rp:b]))
                            s_val = x[int(s_idx)]
                        except Exception:
                            s_idx = np.nan

                    # 4) R' e S' (segundas deflexões relevantes) — janelas ligeiramente mais amplas
                    r2_idx = np.nan
                    s2_idx = np.nan

                    # procura segundo máximo após R (até 100 ms) e exige amplitude >= 15% da de R
                    right_lim = min(b, rp + int(0.10 * fs))
                    if right_lim > rp + 3:
                        (imax2, vmax2), _ = _find_local_extrema(x, rp + 1, right_lim)
                        if np.isfinite(vmax2) and vmax2 > 0.12 * max(vmax, 1e-9):
                            r2_idx = int(imax2)

                    # procura segundo mínimo após S (até 80 ms pós-S) e exige amplitude >= 15% da de S
                    if np.isfinite(s_idx):
                        right_lim2 = min(b, int(s_idx) + int(0.08 * fs))
                        if right_lim2 > s_idx + 3:
                            _, (imin2, vmin2) = _find_local_extrema(x, int(s_idx) + 1, right_lim2)
                            if np.isfinite(vmin2) and abs(vmin2) > 0.12 * max(abs(vmin), 1e-9):
                                s2_idx = int(imin2)

                    # 5) Larguras — tenta primeiro por ENERGIA; se falhar, usa MEIA-AMPLITUDE
                    def _width_ms_energy_then_half(idx, L, R, pol):
                        w = _wave_width_energy(x, idx, L, R, polarity=pol)
                        if not np.isfinite(w):
                            w = _wave_width_half_amp(x, idx, L, R, polarity=pol)
                        return (w / fs) * 1000.0 if np.isfinite(w) else np.nan

                    # Q
                    if np.isfinite(q_idx):
                        q_widths.append(_width_ms_energy_then_half(int(q_idx), a, rp, -1))
                    else:
                        q_widths.append(np.nan)

                    # R
                    r_widths.append(_width_ms_energy_then_half(rp, a, b, +1))

                    # S
                    if np.isfinite(s_idx):
                        s_widths.append(_width_ms_energy_then_half(int(s_idx), rp, b, -1))
                    else:
                        s_widths.append(np.nan)

                    # R'
                    if np.isfinite(r2_idx):
                        r2_widths.append(_width_ms_energy_then_half(int(r2_idx), rp, b, +1))
                    else:
                        r2_widths.append(np.nan)

                    # S'
                    if np.isfinite(s2_idx):
                        s2_widths.append(_width_ms_energy_then_half(int(s2_idx), rp, b, -1))
                    else:
                        s2_widths.append(np.nan)

                    # deflexões/flags (mantém como estava)
                    deflections.append(_count_intrinsic_deflections(x, a, b))
                    ragR.append(_ragged_flag(x, a, b, "R"))
                    dipR.append(_diphasic_flag(x, a, b, "R"))


                to_ms = lambda arr: (np.asarray(arr, dtype=float) / fs) * 1000.0
                f["q_width_ms"]  = _safe_nanmean(to_ms(q_widths))
                f["r_width_ms"]  = _safe_nanmean(to_ms(r_widths))
                f["s_width_ms"]  = _safe_nanmean(to_ms(s_widths))
                f["r2_width_ms"] = _safe_nanmean(to_ms(r2_widths))
                f["s2_width_ms"] = _safe_nanmean(to_ms(s2_widths))
                f["num_intrinsic_defl"] = _safe_nanmean(deflections)
                f["ragged_r"] = _safe_nanmean(ragR)
                f["diphasic_r"] = _safe_nanmean(dipR)

            # ---------- P e T: ragged/diphasic ----------
            pt = _p_t_windows_from_signals(signals)
            if "P" in pt:
                PR = []
                PD = []
                for (a, b) in pt["P"]:
                    PR.append(_ragged_flag(x, a, b, "P"))
                    PD.append(_diphasic_flag(x, a, b, "P"))
                f["ragged_p"]  = _safe_nanmean(PR)
                f["diphasic_p"] = _safe_nanmean(PD)

            if "T" in pt:
                TR = []
                TD = []
                for (a, b) in pt["T"]:
                    TR.append(_ragged_flag(x, a, b, "T"))
                    TD.append(_diphasic_flag(x, a, b, "T"))
                f["ragged_t"]  = _safe_nanmean(TR)
                f["diphasic_t"] = _safe_nanmean(TD)

            f["num_beats"] = float(len(rpeaks)) if rpeaks is not None else np.nan

            # ---- Fallbacks de PR e QT (quando NK não delimitou P/T) ----
            if (not np.isfinite(f["pr_interval_ms"]) or not np.isfinite(f["qt_interval_ms"])):
                if qrs_wins is None and rpeaks is not None and rpeaks.size:
                    qrs_wins = _simple_qrs_windows_from_rpeaks(rpeaks, fs, len(x))
                if qrs_wins is not None and len(qrs_wins):
                    pr_fb, qt_fb = _estimate_pr_qt_from_windows(x, qrs_wins, fs)
                    if not np.isfinite(f["pr_interval_ms"]):
                        f["pr_interval_ms"] = pr_fb
                    if not np.isfinite(f["qt_interval_ms"]):
                        f["qt_interval_ms"] = qt_fb


        except Exception:
            pass  # cai no fallback

    # ---------- Fallback: HR + QRS aproximado ----------
    if not np.isfinite(f["heart_rate_bpm"]):
        try:
            f["heart_rate_bpm"] = _fallback_hr(x, fs)
        except Exception:
            pass

    # tentativa simples de QRS médio por largura de pico maior (quando não veio do NK)
    if not np.isfinite(f["qrs_duration_ms"]):
        try:
            # normaliza
            xn = (x - np.nanmedian(x)) / (np.nanstd(x) + 1e-9)
            thr = max(0.8, np.nanpercentile(xn, 75))
            peaks = np.where((xn[1:-1] > thr) & (xn[1:-1] > xn[:-2]) & (xn[1:-1] > xn[2:]))[0] + 1
            if peaks.size >= 2:
                qrs_wins = _simple_qrs_windows_from_rpeaks(peaks, fs, len(x))
                if qrs_wins is not None and len(qrs_wins):
                    widths = []
                    for a, b in qrs_wins:
                        dur_ms = (b - a) / fs * 1000.0
                        if 60 <= dur_ms <= 160:
                            widths.append(dur_ms)
                    if len(widths):
                        f["qrs_duration_ms"] = float(np.nanmedian(widths))

        except Exception:
            pass

    # tenta primeiro ./models_site/feature_order.json; se não houver, ./feature_order.json
    try:
        import json, os
        from pathlib import Path
        p1 = Path("./models_site/feature_order.json")
        p2 = Path("./feature_order.json")
        if p1.exists():
            order = json.loads(p1.read_text(encoding="utf-8"))
        elif p2.exists():
            order = json.loads(p2.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError
    except Exception:
        # se não existir nenhum dos dois, retorna as labels clínicas cruas
        return {k: (float(v) if v is not None else np.nan) for k, v in f.items()}


    mapping = {
        "f5":  "qrs_duration_ms",
        "f6":  "pr_interval_ms",
        "f7":  "qt_interval_ms",
        "f8":  "t_duration_ms",
        "f9":  "p_duration_ms",
        "f15": "heart_rate_bpm",

        # DII — larguras/flags já calculadas
        "f28": "q_width_ms",
        "f29": "r_width_ms",
        "f30": "s_width_ms",
        "f31": "r2_width_ms",
        "f32": "s2_width_ms",
        "f33": "num_intrinsic_defl",
        "f34": "ragged_r",
        "f35": "diphasic_r",
        "f36": "ragged_p",
        "f37": "diphasic_p",
        "f38": "ragged_t",
        "f39": "diphasic_t",

        # ===== DII específicas (f170–f179) =====
        "f170": "q_width_ms",        # Q wave width (lead II)
        "f171": "r_width_ms",        # R wave width (lead II)
        "f172": "s_width_ms",        # S wave width (lead II)
        "f173": "r2_width_ms",       # R′ wave width (lead II)
        "f174": "s2_width_ms",       # S′ wave width (lead II)
        "f175": "num_intrinsic_defl",# Number of intrinsic deflections (lead II)
        "f176": "ragged_r",          # Ragged R wave (lead II)
        "f177": "diphasic_r",        # Diphasic R wave (lead II)
        "f178": "ragged_p",          # Ragged P wave (lead II)
        "f179": "diphasic_p",        # Diphasic P wave (lead II)
    }


    out = {}
    for key in order:
        if key in mapping:
            val = f.get(mapping[key], np.nan)
        else:
            # se não temos fórmula aqui (ex.: f170..f179), mantemos NaN (não inventar)
            val = np.nan
        out[key] = float(val) if (val is not None and np.isfinite(val)) else np.nan

    return out


def extract_features_from_csv(
    path: str,
    **kwargs,
) -> Dict[str, float]:
    """
    Atalho: lê o CSV do notebook e extrai as features com os filtros padrão.
    Ex:
        feats = extract_features_from_csv("entrada_notebook.csv")
    """
    df = load_notebook_csv(path)
    return extract_features_from_df(df, **kwargs)


# ---------------------------------------
# Helpers internos para a extração
# ---------------------------------------

def _median_duration(signals: pd.DataFrame, on_key: str, off_key: str, fs: float) -> float:
    if on_key not in signals or off_key not in signals:
        return np.nan
    on = np.where(np.diff((signals[on_key].values > 0).astype(int)) == 1)[0]
    off = np.where(np.diff((signals[off_key].values > 0).astype(int)) == 1)[0]
    if len(on) == 0 or len(off) == 0:
        return np.nan
    n = min(len(on), len(off))
    dur = (off[:n] - on[:n]) / fs * 1000.0
    if len(dur) == 0:
        return np.nan
    return float(np.nanmedian(dur))


def _median_interval(signals: pd.DataFrame, a_key: str, b_key: str, fs: float) -> float:
    if a_key not in signals or b_key not in signals:
        return np.nan
    a = np.where(np.diff((signals[a_key].values > 0).astype(int)) == 1)[0]
    b = np.where(np.diff((signals[b_key].values > 0).astype(int)) == 1)[0]
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    interval = (b[:n] - a[:n]) / fs * 1000.0
    if len(interval) == 0:
        return np.nan
    return float(np.nanmedian(interval))


def _fallback_hr(x: np.ndarray, fs: float) -> float:
    if len(x) < int(2 * fs):
        return np.nan
    x = (x - np.nanmedian(x)) / (np.nanstd(x) + 1e-9)
    thr = max(0.8, np.nanpercentile(x, 75))
    peaks = np.where((x[1:-1] > thr) & (x[1:-1] > x[0:-2]) & (x[1:-1] > x[2:]))[0] + 1
    if len(peaks) < 2:
        return np.nan
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.3) & (ibi < 2.5)]
    if len(ibi) == 0:
        return np.nan
    hr = 60.0 / np.nanmedian(ibi)
    return float(hr)

# ===== Helpers adicionais (ondas dentro do QRS, flags, larguras) =====

def _safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else np.nan

def _indices_from_binary_edges(series_bool: np.ndarray):
    """De uma série binária (0/1), retorna arrays de onsets/offsets (subidas/descidas)."""
    b = (series_bool.astype(int) > 0).astype(int)
    up  = np.where(np.diff(b, prepend=0) == 1)[0]
    dn  = np.where(np.diff(b, append=0)  == -1)[0]
    n = min(len(up), len(dn))
    return up[:n], dn[:n]

def _qrs_windows_from_signals(signals: pd.DataFrame, fs: float):
    """Tenta usar onsets/offsets do NK. Se não houver, retorna None para cair no fallback."""
    keys = ["ECG_QRS_Complex_Onsets", "ECG_QRS_Complex_Offsets"]
    if any(k not in signals.columns for k in keys):
        return None
    on_beg, on_end = _indices_from_binary_edges((signals[keys[0]].values > 0))
    off_beg, off_end = _indices_from_binary_edges((signals[keys[1]].values > 0))
    n = min(len(on_beg), len(off_beg))
    if n == 0:
        return None
    return np.vstack([on_beg[:n], off_end[:n]]).T  # (n_beats, 2)

def _simple_qrs_windows_from_rpeaks(rpeaks: np.ndarray, fs: float, n_samples: int):
    """
    Fallback: cria janelas ao redor do R-peak (por ex. -90 ms a +120 ms).
    Garante limites dentro do sinal.
    """
    pre  = int(round(0.07 * fs))   # 90 ms
    post = int(round(0.09 * fs))   # 120 ms
    wins = []
    for rp in rpeaks:
        a = max(0, rp - pre)
        b = min(n_samples - 1, rp + post)
        if b > a + 3:
            wins.append([a, b])
    return np.array(wins, dtype=int) if len(wins) else None

def _wave_width_half_amp(x: np.ndarray, idx_peak: int, left: int, right: int, polarity: int = 1):
    """
    Mede 'largura' de uma onda em torno de um pico, como distância entre
    cruzamentos de meia-amplitude dentro da janela [left, right].
    polarity=+1 para ondas positivas (R, R'), -1 para negativas (Q, S, S').
    Retorna largura em amostras (ou NaN).
    """
    if right <= left or idx_peak <= left or idx_peak >= right:
        return np.nan
    seg = x[left:right+1]
    pk  = x[idx_peak]
    # meia amplitude relativa ao baseline local
    base = np.nanmedian([x[left], x[right]])
    target = base + polarity * 0.5 * (pk - base)
    # procura cruzamentos à esquerda e direita
    left_cross  = np.nan
    right_cross = np.nan
    # esquerda
    for i in range(idx_peak, left, -1):
        if (polarity == 1 and x[i] <= target) or (polarity == -1 and x[i] >= target):
            left_cross = i
            break
    # direita
    for i in range(idx_peak, right):
        if (polarity == 1 and x[i] <= target) or (polarity == -1 and x[i] >= target):
            right_cross = i
            break
    if np.isnan(left_cross) or np.isnan(right_cross) or right_cross <= left_cross:
        return np.nan
    return float(right_cross - left_cross)

def _find_local_extrema(x: np.ndarray, left: int, right: int):
    """Retorna (max_idx, max_val), (min_idx, min_val) dentro da janela."""
    if right <= left:
        return (np.nan, np.nan), (np.nan, np.nan)
    sl = slice(max(0, left), max(0, right)+1)
    seg = x[sl]
    if seg.size < 3:
        return (np.nan, np.nan), (np.nan, np.nan)
    rel_max = np.nanargmax(seg)
    rel_min = np.nanargmin(seg)
    max_idx, min_idx = left + rel_max, left + rel_min
    return (max_idx, x[max_idx]), (min_idx, x[min_idx])

def _count_intrinsic_deflections(x: np.ndarray, left: int, right: int, thr_rel: float = 0.15):
    """
    Número de deflexões intrínsecas no QRS: conta mudanças de concavidade/pequenos picos
    acima de um limiar relativo ao pico principal do complexo.
    """
    if right - left < 5:
        return np.nan
    sl = slice(left, right+1)
    seg = x[sl]
    # normaliza pela amplitude do maior pico absoluto na janela
    A = np.nanmax(np.abs(seg))
    if not np.isfinite(A) or A <= 1e-9:
        return np.nan
    y = seg / A
    # derivada simples
    dy = np.diff(y)
    # pontos onde a derivada cruza zero com magnitude suficiente
    crossings = np.where(np.diff(np.sign(dy)) != 0)[0]
    # filtra por amplitude entre picos
    count = 0
    last = None
    for c in crossings:
        vec = y[max(0, c-1):min(len(y), c+2)]
        if vec.size == 0: 
            continue
        if np.nanmax(np.abs(vec)) >= thr_rel:
            if last is None or (c - last) >= 1:
                count += 1
                last = c
    return float(count)

def _ragged_flag(x: np.ndarray, left: int, right: int, wave: str, thr_rel: float = 0.12):
    """
    'Ragged' = apresenta múltiplos pequenos picos na janela da onda.
    Implementação: se o número de cruzamentos de derivada filtrados > 1.
    """
    cnt = _count_intrinsic_deflections(x, left, right, thr_rel=thr_rel)
    if not np.isfinite(cnt):
        return np.nan
    return float(1.0 if cnt >= 2 else 0.0)

def _diphasic_flag(x: np.ndarray, left: int, right: int, wave: str, thr_rel: float = 0.15):
    """
    'Diphasic' = muda de sinal dentro da janela (ex.: P+/P-).
    Implementação simples: verifica se há max e min relevantes na janela.
    """
    (imax, vmax), (imin, vmin) = _find_local_extrema(x, left, right)
    if not np.isfinite(vmax) or not np.isfinite(vmin):
        return np.nan
    A = max(abs(vmax), abs(vmin))
    if A <= 1e-9:
        return np.nan
    # se ambos têm magnitude relevante e sinais opostos → diphasic
    return float(1.0 if (abs(vmax) >= thr_rel*A and abs(vmin) >= thr_rel*A and np.sign(vmax) != np.sign(vmin)) else 0.0)

def _p_t_windows_from_signals(signals: pd.DataFrame):
    """
    Extrai janelas [onset, offset] de P e T a partir das colunas do NK (se existirem).
    Retorna dict com 'P': array[[on,off],...], 'T': array[[on,off],...]
    """
    out = {}
    for wave, (on_key, off_key) in {
        "P": ("ECG_P_Onsets", "ECG_P_Offsets"),
        "T": ("ECG_T_Onsets", "ECG_T_Offsets"),
    }.items():
        if on_key in signals.columns and off_key in signals.columns:
            on_beg, on_end = _indices_from_binary_edges((signals[on_key].values > 0))
            off_beg, off_end = _indices_from_binary_edges((signals[off_key].values > 0))
            n = min(len(on_beg), len(off_beg))
            if n > 0:
                out[wave] = np.vstack([on_beg[:n], off_end[:n]]).T
    return out

# ===== Fallbacks para PR e QT quando o NK não marcar P/T =====

def _baseline_idx(x: np.ndarray, center: int, span: int, side: str) -> int:
    """
    Retorna o primeiro índice (à esquerda/direita) partindo de 'center' onde o sinal
    cruza/chega perto da mediana local (baseline). Usa janela 'span' em amostras.
    """
    n = len(x)
    if n == 0:
        return np.nan
    lo = max(0, center - span)
    hi = min(n - 1, center + span)
    base = np.nanmedian(x[lo:hi+1])
    if side == "left":
        rg = range(center, lo, -1)
        for i in rg:
            if abs(x[i] - base) <= 0.1 * (np.nanstd(x[lo:hi+1]) + 1e-9):
                return i
        return lo
    else:
        rg = range(center, hi)
        for i in rg:
            if abs(x[i] - base) <= 0.1 * (np.nanstd(x[lo:hi+1]) + 1e-9):
                return i
        return hi


def _estimate_pr_qt_from_windows(x: np.ndarray, qrs_wins: np.ndarray, fs: float) -> Tuple[float, float]:
    """
    Fallback de PR e QT SEM NeuroKit:
      • PR  = (onset da P)  → (onset do QRS)
      • QT  = (onset do QRS)→ (retorno ao baseline após QRS)
    Corrigido para alinhar com valores do dataset UCI:
      PR médio ~160–200 ms, QT ~360–400 ms.
    """
    if qrs_wins is None or len(qrs_wins) == 0 or fs <= 0:
        return np.nan, np.nan

    x = np.asarray(x, dtype=float)
    n = len(x)
    prs, qts = [], []

    # aumentamos as janelas para respeitar fisiologia real
    pre_min = int(round(0.06 * fs))   # 80 ms
    pre_max = int(round(0.32 * fs))   # 280 ms
    post_max = int(round(0.60 * fs))  # 450 ms (para QT completo)

    def _baseline(lo, hi):
        lo = max(0, lo); hi = min(n - 1, hi)
        if hi <= lo: return np.nan
        return float(np.nanmedian(x[lo:hi+1]))

    for (a, b) in qrs_wins:
        a = int(a); b = int(b)
        if a <= 0 or b <= a or b >= n - 1:
            continue

        # ===== PR (P_onset → QRS_onset=a) =====
        L = max(0, a - pre_max)
        R = max(0, a - pre_min)
        if R - L < 5:
            continue

        seg = x[L:R+1]
        base = _baseline(L, R)
        y = seg - base
        dy = np.abs(np.diff(y))

        if len(dy) >= 3:
            k = int(np.nanargmax(dy))  # pico da derivada (subida da P)
            peak_idx = L + k
            lo_win = max(L, peak_idx - int(0.06 * fs))
            hi_win = min(R, peak_idx + int(0.06 * fs))
            sigma = float(np.nanstd(x[lo_win:hi_win+1])) + 1e-9
            thr = 0.10 * sigma
            p_on = np.nan
            for i in range(peak_idx, L, -1):
                if abs(x[i] - base) <= thr:
                    p_on = float(i)
                    break
            if np.isfinite(p_on) and (a - int(p_on)) > int(0.06 * fs):
                prs.append((a - int(p_on)) / fs * 1000.0)

        # ===== QT (QRS_onset=a → baseline após QRS) =====
        lo2 = b
        hi2 = min(n - 1, b + post_max)
        base2 = _baseline(lo2, hi2)
        sigma2 = float(np.nanstd(x[lo2:hi2+1])) + 1e-9
        thr2 = 0.10 * sigma2
        t_off = np.nan
        for j in range(lo2, hi2 + 1):
            if abs(x[j] - base2) <= thr2:
                t_off = float(j)
                break
        if np.isfinite(t_off) and (t_off - a) > int(0.28 * fs):
            qts.append((t_off - a) / fs * 1000.0)

    pr_ms = float(np.nanmedian(prs)) if len(prs) else np.nan
    qt_ms = float(np.nanmedian(qts)) if len(qts) else np.nan
    return pr_ms, qt_ms

def _wave_width_energy(x: np.ndarray, idx_peak: int, left: int, right: int, polarity: int = +1):
    """
    Largura por 'energia' ao redor do pico:
      1) calcula energia cumulativa E[i] = sum((x[i]-baseline)^2)
      2) pega a janela [L,R] cujas energias acumuladas correspondem a 15%..85% da energia total
         dentro de [left,right] em torno do pico.
    Retorna largura em AMOSTRAS (float) ou np.nan.
    """
    if right <= left or not np.isfinite(idx_peak):
        return np.nan
    left = int(max(0, left)); right = int(min(len(x)-1, right))
    if right - left < 4 or idx_peak <= left or idx_peak >= right:
        return np.nan

    seg = x[left:right+1].astype(float)
    base = np.nanmedian([x[left], x[right]])
    y = polarity * (seg - base)  # invertido p/ ondas negativas virar positivo
    # mantém só parte 'acima' do baseline
    y = np.maximum(y, 0.0)
    if np.nanmax(y) <= 1e-12:
        return np.nan

    # energia cumulativa
    e = np.cumsum((y**2))
    e = e - np.nanmin(e)
    Etot = np.nanmax(e)
    if Etot <= 0:
        return np.nan

    # percentis 15%..85%
    e15 = 0.15 * Etot
    e85 = 0.85 * Etot
    # índices relativos à janela
    try:
        iL = int(np.nanargmin(np.abs(e - e15)))
        iR = int(np.nanargmin(np.abs(e - e85)))
    except Exception:
        return np.nan
    if iR <= iL:
        return np.nan
    # volta a índices absolutos
    L = left + iL
    R = left + iR
    return float(R - L)

# ==========================================
# 7) EXEMPLO RÁPIDO
# ==========================================

def example_parse_lines_to_csv(lines: Iterable[str], csv_path: str) -> pd.DataFrame:
    acc = RawAccumulator()
    for ln in lines:
        acc.add_line(ln)
    acc.to_csv(csv_path)
    return acc.get_df()





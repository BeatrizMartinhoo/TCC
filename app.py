# ==========================================
# BIBLIOTECAS
# ==========================================
from __future__ import annotations

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # evita variações e stalls raros
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # logs mais silenciosos


import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# === IA: carregar uma única vez por sessão ===
@st.cache_resource(show_spinner=False)
def get_predictor():
    # Import tardio evita carregar TF/Keras em todo rerun
    from arrhythmia_infer import predict_from_features
    return predict_from_features


from ecg_components import (
    init_session_state, list_serial_ports, connect_to_serial, disconnect_from_serial,
    start_collection, render_live_plot_and_metrics,
    get_notebook_df, make_ecg_plot_png, build_docx_report
)

from ecg_features import filter_df_like_notebook, extract_features_from_df, infer_fs_from_tms


# ==========================================
# 🔹 RÓTULOS DAS FEATURES (fixo no app)
# ==========================================
FEATURE_LABELS = {
    "f5":  "qrs_duration_ms",
    "f6":  "pr_interval_ms",
    "f7":  "qt_interval_ms",
    "f8":  "t_duration_ms",
    "f9":  "p_duration_ms",
    "f15": "heart_rate_bpm",
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
    # DII 170–179
    "f170": "q_width_ms",
    "f171": "r_width_ms",
    "f172": "s_width_ms",
    "f173": "r2_width_ms",
    "f174": "s2_width_ms",
    "f175": "num_intrinsic_defl",
    "f176": "ragged_r",
    "f177": "diphasic_r",
    "f178": "ragged_p",
    "f179": "diphasic_p",
}

# ==========================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Wearable ECG – IA para Arritmias",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="collapsed"
)

# Inicializa estado
init_session_state()

# Artefatos de IA — ordem das features (se existir)
MODELS_DIR = Path("./models_site")
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"

def _load_feature_order() -> list:
    if FEATURE_ORDER_PATH.exists():
        return json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    local_json = Path("./feature_order.json")
    if local_json.exists():
        return json.loads(local_json.read_text(encoding="utf-8"))
    return []  # segue sem ordem fixa

def _build_features_row(feature_order: list) -> dict:
    row = {k: np.nan for k in feature_order}
    bpm = st.session_state.get("bpm", None)
    for key in feature_order:
        if key.strip().lower() in ["heart rate (bpm)", "heart_rate_bpm", "hr_bpm", "f15"]:
            if bpm is not None:
                row[key] = float(bpm)
            break
    return row

# ==========================================
# CABEÇALHO
# ==========================================
st.markdown(
    """
    <div style="text-align:center; padding-top: 1.5rem; padding-bottom: 1rem;">
        <h1 style="color:#000000; font-size:2.4rem; margin-bottom:0.2rem;">
            🫀 Wearable de ECG para Identificação de Arritmias com IA
        </h1>
        <p style="color:#bbbbbb; font-size:1.0rem; margin-top:0.4rem;">
            Trabalho de Conclusão de Curso (TCC) • Beatriz Luísie • PUC-SP • 2025
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# DADOS DO PACIENTE (robusto)
# ==========================================
st.subheader("Dados do paciente")

def safe_int(x):
    try:
        xi = int(float(x))
        return xi if xi >= 0 else None
    except Exception:
        return None

def safe_float(x):
    try:
        xf = float(x)
        return xf if xf >= 0 else None
    except Exception:
        return None

if "patient" not in st.session_state or not isinstance(st.session_state.patient, dict):
    st.session_state.patient = {
        "nome": "",
        "idade": None,
        "genero": "Feminino",
        "altura_cm": None,
        "peso_kg": None,
    }

genero_opts = ["Feminino", "Masculino", "Outro/Prefiro não dizer"]
_gen_value = st.session_state.patient.get("genero", "Feminino")
try:
    _gen_index = genero_opts.index(_gen_value)
except ValueError:
    _gen_index = 0

# ======= Mensagens do formulário (fora do form) =======
form_msg = st.empty()  # placeholder para mensagens de sucesso/erro

# ======= Formulário do paciente =======
# ======= Mensagens do formulário (fora do form) =======
form_msg = st.empty()  # placeholder para mensagens

# ======= Formulário do paciente (layout: Nome+Gênero / Idade+Altura+Peso) =======
with st.form("patient_form", clear_on_submit=False):
    # valores atuais salvos (podem ser None)
    pac = st.session_state.get("patient", {}) or {}
    nome_cur  = pac.get("nome") or ""
    idade_cur = pac.get("idade")
    alt_cur   = pac.get("altura_cm")
    peso_cur  = pac.get("peso_kg")
    gen_cur   = pac.get("genero") or "Feminino"

    # defaults SEGUROS (v1.43 não aceita None em number_input)
    idade_default = int(idade_cur) if isinstance(idade_cur, (int, float)) else 0
    alt_default   = int(alt_cur)   if isinstance(alt_cur, (int, float))   else 0
    peso_default  = float(peso_cur) if isinstance(peso_cur, (int, float)) else 0.0

    # --- Linha 1: Nome (2) | Gênero (1)
    row1 = st.columns([2, 1])
    with row1[0]:
        nome = st.text_input("Nome", value=nome_cur)
    with row1[1]:
        genero_opts = ["Feminino", "Masculino", "Outro/Prefiro não dizer"]
        try:
            gen_index = genero_opts.index(gen_cur)
        except ValueError:
            gen_index = 0
        genero = st.selectbox("Gênero", options=genero_opts, index=gen_index)

    # --- Linha 2: Idade (1) | Altura (1) | Peso (1)
    row2 = st.columns(3)
    with row2[0]:
        idade = st.number_input("Idade (anos)", min_value=0, max_value=120, step=1, value=idade_default)
    with row2[1]:
        altura_cm = st.number_input("Altura (cm)", min_value=0, max_value=300, step=1, value=alt_default)
    with row2[2]:
        peso_kg = st.number_input("Peso (kg)", min_value=0.0, max_value=400.0, step=0.1, value=peso_default)

    submitted = st.form_submit_button("Salvar dados", use_container_width=True)

# ======= Pós-submit (FORA do with st.form) =======
if submitted:
    try:
        st.session_state.patient = {
            "nome": (nome or "").strip(),
            "idade": int(idade) if idade is not None else 0,
            "altura_cm": int(altura_cm) if altura_cm is not None else 0,
            "peso_kg": float(peso_kg) if peso_kg is not None else 0.0,
            "genero": str(genero),
        }
        form_msg.success("Dados do paciente salvos nesta sessão.")
    except Exception as e:
        form_msg.error(f"Falha ao salvar dados do paciente: {e}")




col_clear, _ = st.columns([1, 3])
with col_clear:
    if st.button("Limpar dados do paciente"):
        st.session_state.patient = {
            "nome": "",
            "idade": None,
            "genero": "Feminino",
            "altura_cm": None,
            "peso_kg": None,
        }
        st.info("Dados de paciente limpos desta sessão.")


# ==========================================
# SIDEBAR DE CONEXÃO SERIAL
# ==========================================
with st.sidebar:
    with st.expander("⚙️ Conexão Bluetooth", expanded=True):
        ports = list_serial_ports()
        default_port = st.session_state.get("selected_port", ports[0] if ports else "")

        if ports:
            port = st.selectbox(
                "Porta serial",
                ports,
                index=ports.index(default_port) if default_port in ports else 0,
                placeholder="Selecione a COM",
                label_visibility="collapsed"
            )
        else:
            port = ""
            st.info("Nenhuma porta encontrada. Conecte a ESP32 e clique em **R** para recarregar a página.")

        baud = st.selectbox("Baudrate", [115200, 57600, 38400, 9600], index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔗 Conectar", type="primary",
                        disabled=st.session_state.get("connected", False) or not ports):
                ser, msg = connect_to_serial(port, baud)
                st.session_state.ser = ser
                st.session_state.connected = ser is not None
                if ser:
                    st.session_state.selected_port = port
                st.session_state["msg_status"] = msg

        with col_b:
            if st.button("Desconectar", disabled=not st.session_state.get("connected", False)):
                msg = disconnect_from_serial()
                st.session_state.connected = False
                st.session_state["msg_status"] = msg

        msg_ph = st.empty()
        msg_text = st.session_state.get("msg_status", "")
        if msg_text:
            msg_lower = msg_text.lower()
            if any(w in msg_lower for w in ["conectado", "sucesso", "iniciado"]):
                msg_ph.success(msg_text)
            elif any(w in msg_lower for w in ["falha", "erro", "não", "nao"]):
                msg_ph.error(msg_text)
            else:
                msg_ph.info(msg_text)
    st.divider()

# ==========================================
# COLETA + TRAÇADO
# ==========================================
st.divider()
st.subheader("Coleta e traçado em tempo real")

col_left, col_right = st.columns([1, 4], gap="large")
with col_right:
    plot_ph = st.empty()
with col_left:
    st.markdown("#### Controles")
    start_disabled = (not st.session_state.get("connected", False)
                      or st.session_state.get("collecting", False))
    clicked = st.button("Iniciar coleta (30 s)", type="primary",
                    use_container_width=True, disabled=start_disabled)



    if "time_ph" not in st.session_state:
        st.session_state.time_ph = st.empty()
    time_ph = st.session_state.time_ph
    time_ph.caption("⏱️ Tempo decorrido: 0.0 s")

    if "bpm_ph" not in st.session_state:
        st.session_state.bpm_ph = st.empty()
    bpm_ph = st.session_state.bpm_ph

    if clicked:
        start_collection(30.0, plot_ph=plot_ph, bpm_ph=bpm_ph, console_ph=None)

if not st.session_state.get("collecting", False):
    render_live_plot_and_metrics(plot_ph, bpm_ph, None)

# ==========================================
# 🔧 PRÉ-PROCESSAMENTO
# ==========================================
st.divider()
st.subheader("🔧 Pré-processamento do sinal")

with st.container(border=True):
    st.markdown("Clique abaixo para gerar o arquivo do exame, aplicar os filtros e capturar as features automaticamente.")
    run_all = st.button("▶️ Iniciar pré-processamento", type="primary", use_container_width=True)

    if run_all:
        try:
            df_nb = get_notebook_df(clear=False)
            if df_nb.empty:
                st.warning("Nenhum dado RAW disponível. Realize uma coleta primeiro.")
            else:
                fs = infer_fs_from_tms(df_nb["t_ms"])
                if fs is None:
                    st.warning("Não foi possível inferir fs a partir de t_ms.")
                else:
                    df_f = filter_df_like_notebook(
                        df_nb,
                        fs=fs,
                        signal_col="value",
                        out_col="ecg_proc",
                        assume_is_adc=True,
                        respect_lo_mask=True,
                        notch_f0_hz=60.0, notch_Q=30.0,
                        bp_low_hz=0.7, bp_high_hz=40.0, bp_order=3,
                        sg_win_s=0.12, sg_poly=3,
                    )
                    feats = extract_features_from_df(
                        df_f,
                        fs=fs,
                        signal_col="ecg_proc",
                        apply_filters=False,
                        filtered_col_out="ecg_proc",
                        assume_is_adc=False,
                        use_neurokit=True,
                    )
                    st.success("✅ Features captadas com sucesso a partir do sinal pré-processado!")

                    feats_final = {}
                    for fkey, label in FEATURE_LABELS.items():
                        v = feats.get(label, np.nan)
                        feats_final[fkey] = float(v) if isinstance(v, (int, float, np.floating)) else np.nan
                    for k, v in feats.items():
                        if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
                            feats_final[k] = float(v) if isinstance(v, (int, float, np.floating)) else np.nan

                    st.session_state["last_features_dict"] = feats_final
        except Exception as e:
            st.error(f"Falha no pré-processamento: {e}")

# ==========================================
# 🤖 CLASSIFICAÇÃO COM IA
# ==========================================
st.divider()
st.subheader("🤖 Classificação com IA")

with st.container(border=True):
    st.markdown("Clique para enviar as features captadas e obter a classificação pela IA treinada.")
    run_infer = st.button("📤 Enviar para IA", type="primary", use_container_width=True)


    if run_infer:
        try:
            feats = st.session_state.get("last_features_dict", {})
            if not feats:
                st.warning("Nenhuma feature captada. Execute o pré-processamento primeiro.")
            else:
                #st.write("📋 Features enviadas à IA:")
                #st.json(feats)

                # usa o preditor cacheado (carregado 1x por sessão)
                predict_from_features = get_predictor()
                result = predict_from_features(feats, method="stacking", thr_override=0.75)

                st.session_state["last_ai_result"] = result  # para o DOCX

                label = result["label"]
                prob = result["score"] * 100
                thr = result["threshold"] * 100

                if label == 1:
                    st.success(f"✅ Resultado: **ARRITMIA DETECTADA** (probabilidade {prob:.1f}% ≥ limiar {thr:.1f}%)")
                else:
                    st.info(f"🫀 Resultado: **SINAL NORMAL** (probabilidade {prob:.1f}% < limiar {thr:.1f}%)")
        except Exception as e:
            st.error(f"Erro ao enviar para a IA: {e}")



# ==========================================
# RELATÓRIO DOCX
# ==========================================
st.divider()
st.subheader("Exportar relatório")

# estado para os bytes / nome do arquivo
if "docx_bytes" not in st.session_state:
    st.session_state.docx_bytes = None
if "docx_fname" not in st.session_state:
    st.session_state.docx_fname = None

colE1, colE2 = st.columns([1,1])
with colE1:
    modo = st.radio(
        "Qual traçado incluir?",
        options=["Janela atual (~10 s)", "Coleta completa (até 30 s)"],
        index=0,
        horizontal=True
    )
    mode_key = "window" if "Janela" in modo else "all"

with colE2:
    if st.button("Gerar relatório (DOCX)", key="btn_make_docx", type="primary", use_container_width=True):
        try:
            docx_bytes = build_docx_report(st.session_state.get("patient", {}), mode=mode_key)
            if not docx_bytes:
                st.warning("Não foi possível gerar o DOCX (verifique dados/instalações).")
            else:
                # nome de arquivo seguro: usa nome do paciente (se houver) + data/hora
                patient = st.session_state.get("patient", {}) or {}
                nome = (patient.get("nome") or "").strip()
                safe_nome = "".join(ch for ch in nome if ch.isalnum() or ch in (" ", "_", "-")).strip()
                safe_nome = safe_nome.replace(" ", "_") if safe_nome else "relatorio_ecg"
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                fname = f"{safe_nome}_{ts}.docx"

                st.session_state.docx_bytes = docx_bytes
                st.session_state.docx_fname = fname
                st.success("Relatório gerado. Use o botão abaixo para baixar.")
        except Exception as e:
            st.error(f"Falha ao gerar DOCX: {e}")

# botão de download persistente (compatível com 1.43)
if st.session_state.docx_bytes and st.session_state.docx_fname:
    st.download_button(
        "Baixar relatório.docx",
        data=st.session_state.docx_bytes,
        file_name=st.session_state.docx_fname,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key="download_docx_fixed",
        use_container_width=True,
    )
else:
    st.caption("Gere o relatório para habilitar o download.")

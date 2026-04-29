import streamlit as st
import torch
import numpy as np
import wfdb
import tempfile
import os
import matplotlib.pyplot as plt
from src.model import CNN1D

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
DESCRIPTIONS = {
    'NORM': 'Normal — aucune anomalie détectée',
    'MI':   'Infarctus du myocarde',
    'STTC': 'Anomalie ST/T (ischémie possible)',
    'CD':   'Trouble de la conduction',
    'HYP':  'Hypertrophie cardiaque'
}

@st.cache_resource
def load_model():
    model = CNN1D(n_leads=12, n_classes=5)
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
    model.eval()
    return model

def preprocess_signal(signal):
    """Normalisation identique à l'entraînement."""
    signal = signal.T.astype(np.float32)  # (12, 1000)
    mean = signal.mean(axis=1, keepdims=True)
    std  = signal.std(axis=1, keepdims=True) + 1e-8
    return (signal - mean) / std

def plot_ecg(signal):
    """Affiche les 12 dérivations ECG."""
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(12, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(signal[:, i], linewidth=0.8, color='#e74c3c')
        ax.set_ylabel(leads[i], fontsize=7, rotation=0, labelpad=20)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[-1].set_xlabel("Échantillons (100 Hz)")
    plt.tight_layout()
    return fig

# --- Interface ---
st.title("🫀 Classificateur ECG — PTB-XL")
st.markdown("Uploadez un fichier ECG au format WFDB (`.dat` + `.hea`) pour obtenir une prédiction.")

col1, col2 = st.columns(2)
dat_file = col1.file_uploader("Fichier .dat", type=['dat'])
hea_file = col2.file_uploader("Fichier .hea", type=['hea'])

if dat_file and hea_file:
    # Sauvegarder dans un dossier fixe
    tmp_dir = '/tmp/ecg_upload'
    os.makedirs(tmp_dir, exist_ok=True)

    dat_name = dat_file.name
    hea_name = hea_file.name
    record_name = dat_name.replace('.dat', '')

    with open(os.path.join(tmp_dir, dat_name), 'wb') as f:
        f.write(dat_file.read())
    with open(os.path.join(tmp_dir, hea_name), 'wb') as f:
        f.write(hea_file.read())

    signal, _ = wfdb.rdsamp(os.path.join(tmp_dir, record_name))

    # Visualiser
    st.subheader("Signal ECG brut")
    st.pyplot(plot_ecg(signal))

    # Prédiction
    model = load_model()
    x = preprocess_signal(signal)
    x = torch.tensor(x).unsqueeze(0)  # (1, 12, 1000)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx   = probs.argmax().item()
    pred_class = SUPERCLASSES[pred_idx]
    confidence = probs[pred_idx].item()

    # Afficher le résultat
    st.subheader("Résultat")
    if pred_class == 'NORM':
        st.success(f"**{pred_class}** — {DESCRIPTIONS[pred_class]} ({confidence:.1%})")
    else:
        st.warning(f"**{pred_class}** — {DESCRIPTIONS[pred_class]} ({confidence:.1%})")

    # Graphique des probabilités
    st.subheader("Probabilités par classe")
    fig2, ax = plt.subplots(figsize=(8, 3))
    colors = ['#2ecc71' if c == pred_class else '#3498db' for c in SUPERCLASSES]
    ax.barh(SUPERCLASSES, probs.numpy(), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité")
    st.pyplot(fig2)
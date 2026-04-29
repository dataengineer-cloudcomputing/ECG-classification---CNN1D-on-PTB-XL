# ECG Classification — CNN 1D on PTB-XL

Classification automatique d'électrocardiogrammes en 5 catégories diagnostiques
à l'aide d'un réseau de neurones convolutif 1D entraîné sur le dataset PTB-XL.

## Problème

Un ECG est un signal temporel capté sur 12 dérivations pendant 10 secondes.
L'objectif est de classifier chaque ECG dans l'une des 5 superclasses :

| Classe | Description |
|--------|-------------|
| NORM   | ECG normal |
| MI     | Infarctus du myocarde |
| STTC   | Anomalie ST/T (ischémie possible) |
| CD     | Trouble de la conduction |
| HYP    | Hypertrophie cardiaque |

## Dataset

**PTB-XL** (PhysioNet, 2022) — https://physionet.org/content/ptb-xl/1.0.3/
- 21 799 ECGs de 10 secondes
- 12 dérivations, 100 Hz
- Split officiel : folds 1-8 train / fold 9 val / fold 10 test

## Architecture

CNN 1D à 3 blocs convolutifs :
Input (batch, 12, 1000)
→ Conv1D(12→32, k=7) + BatchNorm + ReLU + MaxPool  → (batch, 32, 500)
→ Conv1D(32→64, k=5) + BatchNorm + ReLU + MaxPool  → (batch, 64, 250)
→ Conv1D(64→128, k=5) + BatchNorm + ReLU + MaxPool → (batch, 128, 125)
→ GlobalAvgPool                                     → (batch, 128)
→ Linear(128→64) + ReLU + Dropout(0.5)
→ Linear(64→5)

## Utilisation rapide après clonage

# 1. Cloner le projet
git clone https://github.com/dataengineer-cloudcomputing/ECG-classification---CNN1D-on-PTB-XL.git
cd ECG-classification---CNN1D-on-PTB-XL

# 2. Installer les dépendances
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Télécharger les données
aws s3 sync --no-sign-request \
  s3://physionet-open/ptb-xl/1.0.3/records100/ ./data/ptb-xl/records100/
aws s3 cp --no-sign-request \
  s3://physionet-open/ptb-xl/1.0.3/ptbxl_database.csv ./data/ptb-xl/
aws s3 cp --no-sign-request \
  s3://physionet-open/ptb-xl/1.0.3/scp_statements.csv ./data/ptb-xl/

# 4. Lancer la démo
streamlit run app.py
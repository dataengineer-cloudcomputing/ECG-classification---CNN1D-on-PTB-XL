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
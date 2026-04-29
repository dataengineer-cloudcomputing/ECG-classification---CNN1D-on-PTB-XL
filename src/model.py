import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    CNN 1D pour classification ECG.

    Entrée  : (batch, 12, 1000) — 12 dérivations, 1000 points temporels
    Sortie  : (batch, 5)        — scores pour les 5 superclasses

    Pourquoi CNN 1D ?
    - Le signal ECG est une série temporelle 1D par dérivation
    - Les convolutions détectent des patterns locaux (complexe QRS, onde P, onde T)
      quelle que soit leur position dans le signal → invariance temporelle
    - Plus simple à entraîner qu'un LSTM, plus interprétable à l'oral
    """

    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()

        # Bloc 1 : capturer les patterns fins (complexe QRS ~50ms)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_leads, out_channels=32,
                      kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # 1000 → 500
        )

        # Bloc 2 : patterns à moyenne échelle (intervalle RR)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)               # 500 → 250
        )

        # Bloc 3 : patterns globaux (morphologie générale)
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)               # 250 → 125
        )

        # Pooling global : résume chaque feature map en 1 valeur
        # → indépendant de la longueur du signal
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
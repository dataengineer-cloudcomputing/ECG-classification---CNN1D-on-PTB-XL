import os
import ast
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset

# Les 5 superclasses officielles du benchmark PTB-XL
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def load_metadata(path):
    """
    Charge ptbxl_database.csv et scp_statements.csv.
    Retourne le dataframe avec une colonne 'label' (int 0-4).
    """
    df = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'),
                     index_col='ecg_id')

    # scp_codes est stocké comme string, on le convertit en dict Python
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    # Charger les descriptions des codes SCP pour connaître la superclasse
    scp = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)

    def get_superclass(scp_codes_dict):
        """
        Pour un ECG donné, on cherche le code SCP dont la superclasse
        est dans nos 5 classes, avec la confiance la plus haute.
        """
        best_class = None
        best_conf = -1
        for code, conf in scp_codes_dict.items():
            if code in scp.index:
                sc = scp.loc[code, 'diagnostic_class']
                if sc in SUPERCLASSES and conf > best_conf:
                    best_class = sc
                    best_conf = conf
        return best_class

    df['superclass'] = df['scp_codes'].apply(get_superclass)

    # Garder uniquement les ECG avec une superclass connue
    df = df.dropna(subset=['superclass'])

    # Encoder en entier (NORM=0, MI=1, STTC=2, CD=3, HYP=4)
    df['label'] = df['superclass'].apply(SUPERCLASSES.index)

    return df


class ECGDataset(Dataset):
    """
    Dataset PyTorch pour PTB-XL.
    Chaque item retourne :
      - signal : tensor float32 de shape (12, 1000)
                 12 dérivations, 1000 points à 100 Hz = 10 secondes
      - label  : tensor long (0 à 4)
    """

    def __init__(self, df, path, normalize=True):
        self.df = df.reset_index()
        self.path = path
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Lire le signal brut avec wfdb
        # filename_lr = chemin relatif vers le fichier sans extension
        record_path = os.path.join(self.path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(record_path)
        # signal shape : (1000, 12) → on transpose en (12, 1000)
        # pour que les convolutions 1D s'appliquent sur la dimension temporelle
        signal = signal.T.astype(np.float32)

        # Normalisation par dérivation (zero mean, unit variance)
        if self.normalize:
            mean = signal.mean(axis=1, keepdims=True)
            std  = signal.std(axis=1, keepdims=True) + 1e-8
            signal = (signal - mean) / std

        return torch.tensor(signal), torch.tensor(row['label'], dtype=torch.long)


def get_splits(df):
    """
    Split officiel PTB-XL :
    strat_fold 1-8 = train, 9 = val, 10 = test
    """
    train = df[df['strat_fold'] <= 8]
    val   = df[df['strat_fold'] == 9]
    test  = df[df['strat_fold'] == 10]
    return train, val, test
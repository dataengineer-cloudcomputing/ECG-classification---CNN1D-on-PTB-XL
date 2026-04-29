import sys
sys.path.insert(0, '.')

from src.dataset import load_metadata, ECGDataset, get_splits
from torch.utils.data import DataLoader

PATH = './data/ptb-xl/'

# 1. Charger les métadonnées
print("Chargement des métadonnées...")
df = load_metadata(PATH)
print(f"ECGs total : {len(df)}")
print(f"Distribution des classes :\n{df['superclass'].value_counts()}")

# 2. Splitter
train_df, val_df, test_df = get_splits(df)
print(f"\nTrain : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

# 3. Tester le Dataset sur 1 sample
dataset = ECGDataset(train_df, PATH)
signal, label = dataset[0]
print(f"\nSignal shape : {signal.shape}")  # attendu : torch.Size([12, 1000])
print(f"Label : {label.item()}")

print("\nTout fonctionne ✓")
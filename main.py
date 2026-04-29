import sys
sys.path.insert(0, '.')

from src.dataset import load_metadata, get_splits
from src.model import CNN1D
from src.train import run_training

PATH = './data/ptb-xl/'

df = load_metadata(PATH)
train_df, val_df, test_df = get_splits(df)

model = CNN1D(n_leads=12, n_classes=5)

run_training(
    model=model,
    train_df=train_df,
    val_df=val_df,
    path=PATH,
    n_epochs=20,
    batch_size=64,
    lr=1e-3
)

from src.train import evaluate
from src.dataset import ECGDataset
from torch.utils.data import DataLoader
import torch

model.load_state_dict(torch.load('best_model.pt'))
device = torch.device('cpu')
test_dataset = ECGDataset(test_df, PATH)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
_, test_acc, test_f1 = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
print(f"Test Acc: {test_acc:.3f} | Test F1: {test_f1:.3f}")
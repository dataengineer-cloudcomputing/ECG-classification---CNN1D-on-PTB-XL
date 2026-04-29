import sys
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader
from src.dataset import load_metadata, get_splits, ECGDataset
from src.model import CNN1D
from src.train import evaluate

PATH = './data/ptb-xl/'

df = load_metadata(PATH)
_, _, test_df = get_splits(df)

model = CNN1D(n_leads=12, n_classes=5)
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
device = torch.device('cpu')
model = model.to(device)

test_dataset = ECGDataset(test_df, PATH)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

_, test_acc, test_f1 = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
print(f"Test Acc: {test_acc:.3f} | Test F1: {test_f1:.3f}")
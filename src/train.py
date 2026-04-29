import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Une époque d'entraînement.
    Retourne la loss moyenne et l'accuracy sur le train set.
    """
    model.train()   # mode entraînement (active Dropout, BatchNorm en mode train)
    total_loss = 0
    correct = 0
    total = 0

    for signals, labels in loader:
        # Déplacer les données sur GPU si disponible
        signals = signals.to(device)   # (batch, 12, 1000)
        labels  = labels.to(device)    # (batch,)

        # 1. Remise à zéro des gradients (obligatoire à chaque batch)
        optimizer.zero_grad()

        # 2. Forward pass : le modèle fait sa prédiction
        outputs = model(signals)       # (batch, 5)

        # 3. Calcul de la loss
        loss = criterion(outputs, labels)

        # 4. Backward pass : calcul des gradients
        loss.backward()

        # 5. Mise à jour des poids
        optimizer.step()

        # Statistiques
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct   += (predicted == labels).sum().item()
        total     += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """
    Évaluation sur val ou test set.
    Retourne loss, accuracy et F1-score macro.
    """
    model.eval()    # mode évaluation (désactive Dropout, BatchNorm en mode eval)
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():   # pas de calcul de gradient en éval
        for signals, labels in loader:
            signals = signals.to(device)
            labels  = labels.to(device)

            outputs   = model(signals)
            loss      = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average='macro')

    return total_loss / len(loader), acc, f1


def run_training(model, train_df, val_df, path, n_epochs=20, batch_size=64, lr=1e-3):
    """
    Boucle d'entraînement complète avec early stopping.
    """
    from src.dataset import ECGDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entraînement sur : {device}")

    # Datasets et DataLoaders
    train_dataset = ECGDataset(train_df, path)
    val_dataset   = ECGDataset(val_df,   path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                          shuffle=False, num_workers=0)

    model = model.to(device)

    # CrossEntropy car classification multi-classe
    # On ajoute class_weight pour compenser le déséquilibre des classes
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Réduire le LR si la val loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 5

    for epoch in range(n_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")

        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("  → Meilleur modèle sauvegardé")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping à l'époque {epoch+1}")
                break

    print("Entraînement terminé.")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for branch_batch, trunk_batch, target_batch in dataloader:
        branch_batch = [b.to(device) for b in branch_batch]
        trunk_batch = trunk_batch.to(device)
        target_batch = target_batch.to(device)

        output = model(branch_batch, trunk_batch)
        loss = criterion(output.squeeze(-1), target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for branch_batch, trunk_batch, target_batch in dataloader:
            branch_batch = [b.to(device) for b in branch_batch]
            trunk_batch = trunk_batch.to(device)
            target_batch = target_batch.to(device)

            output = model(branch_batch, trunk_batch)
            loss = criterion(output.squeeze(-1), target_batch)
            val_loss += loss.item()
    return val_loss / len(dataloader)

def train_model(
    model, dataset, device='cuda', num_epochs=500, batch_size=64, 
    lr=0.001, patience=10, val_ratio=0.2, k_fold=None, multi_gpu=False,
    working_dir="experiment"
):
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    if k_fold:
        # K-Fold Cross-Validation
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\n=== Fold {fold+1}/{k_fold} ===")
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)

            best_val_loss = float('inf')
            epochs_no_improve = 0
            save_path = os.path.join(checkpoint_dir, f"best_model_fold{fold+1}.pt")

            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = evaluate(model, val_loader, criterion, device)

                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping triggered.")
                        break
            fold_results.append(best_val_loss)
        print(f"\nCV Results: Mean={np.mean(fold_results):.6f}, Std={np.std(fold_results):.6f}")
        return fold_results

    else:
        # Default: train/val split
        val_len = int(len(dataset) * val_ratio)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        save_path = os.path.join(checkpoint_dir, "best_model.pt")

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break
        return train_losses, val_losses

import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(X, y):
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    #print(f"X shape: {X_tensor.shape}")
    #print(f"y shape: {y_tensor.shape}")

    # Split into train/validation
    split_idx = int(0.9 * len(X_tensor))
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

    #print(f"Training samples: {len(X_train)}")
    #print(f"Validation samples: {len(X_val)}")

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch.save(X_tensor, "src\\cache\\X_tensor.pth")
    torch.save(y_tensor, "src\\cache\\y_tensor.pth")
    
    return train_loader, val_loader
    #print(f"Batch size: {batch_size}")
    #print(f"Training batches: {len(train_loader)}")
    #print(f"Validation batches: {len(val_loader)}")
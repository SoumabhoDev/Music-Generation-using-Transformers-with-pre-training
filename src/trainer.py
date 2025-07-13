import torch
import torch.nn as nn
import torch.optim as optim
from src.load_dataset import device
# Training configuration
#criterion = nn.MSELoss()
#optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Training loop
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs, targets
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Use only the last token's output
        last_output = outputs[:, -1, :]
        loss = criterion(last_output, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            last_output = outputs[:, -1, :]
            loss = criterion(last_output, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
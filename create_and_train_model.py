import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import json

from src.midi_loader import load_midi_files
from src.build_dataset import create_training_data
from src.load_dataset import load_dataset, device
from src.model_architecture import MusicTransformer
from src.trainer import train_epoch, validate

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open("params.json", encoding='utf-8') as f:
    js = json.load(f)
hparams = js['hparams']

midi_files = load_midi_files(js['DATASET_DIRECTORY'])
X, y = create_training_data(midi_files)
train_loader, val_loader = load_dataset(X, y)

model = MusicTransformer(
    input_dim=4,
    d_model=hparams['d_model'],
    n_heads=hparams['n_heads'],
    n_layers=hparams['n_layers'],
    d_ff=hparams['d_ff'],
    max_seq_len=js['CONTEXT_LENGTH'],
    dropout=hparams['dropout']
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# Training
num_epochs = 16
train_losses = []
val_losses = []
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Training
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    
    # Validation
    val_loss = validate(model, val_loader, criterion)
    
    # Update learning rate
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    #print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_music_transformer.pth')
        print("Saved best model!")

print("Training completed!")

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()




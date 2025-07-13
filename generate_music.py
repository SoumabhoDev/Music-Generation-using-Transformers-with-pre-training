import json
import torch
from src.model_architecture import MusicTransformer
from src.load_dataset import device

with open("params.json") as f:
    js = json.load(f)
X = torch.load("src\\cache_impForGeneration\\X_tensor.pth")

hparams = js['hparams']

model = MusicTransformer(
    input_dim=4,
    d_model=hparams['d_model'],
    n_heads=hparams['n_heads'],
    n_layers=hparams['n_layers'],
    d_ff=hparams['d_ff'],
    max_seq_len=js['CONTEXT_LENGTH'],
    dropout=hparams['dropout']
).to(device)

model.load_state_dict(torch.load(js['BEST_MODEL_PATH'], weights_only=False, map_location=device))

import numpy.random as random
from src.generate import generate_music, tokens_to_midi
# Generate music samples
print("Generating music samples...")

# Use a seed from training data
seed_idx = random.randint(0, len(X) - 1)
seed_sequence = X[seed_idx]

print(f"Using seed sequence from index {seed_idx}")

# Generate with different temperatures
temperatures = [0, 0.2, 0.5]# 0.5, 0.8, 1.0, 1.2
n_tokens = 1000
for temp in temperatures:
    print(f"\nGenerating with temperature {temp}...")
    
    generated_tokens = generate_music(
        model, 
        seed_sequence, 
        num_steps=n_tokens, 
        temperature=temp
    )
    
    output_file = f'sample-results\\generated_music_temp_{temp}_16epc{n_tokens}tokens.mid'
    tokens_to_midi(generated_tokens.cpu().numpy().tolist(), output_file)
    
    print(f"Generated {len(generated_tokens)} tokens")

print("\nGeneration completed! Check the generated MIDI files.")
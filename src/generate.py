import torch
from src.load_dataset import device
import numpy as np
import miditoolkit
import json
with open(".\\params.json", encoding='utf-8') as f:
    js = json.load(f)
def generate_music(model, seed_sequence, num_steps=50, temperature=0.8):
    """Generate music using the trained model with temperature sampling"""
    model.eval()
    
    # Start with seed sequence
    generated = torch.clone(seed_sequence)
    
    with torch.no_grad():
        for _ in range(num_steps):
            # Take last context_length tokens
            context = generated[-js['CONTEXT_LENGTH']:]
            
            # Pad if necessary
            while len(context) < js['CONTEXT_LENGTH']:
                context = [[-1.0, -1.0, -1.0, -1.0]] + context
            
            # Convert to tensor
            context_tensor = context.unsqueeze(0).float().to(device)#torch.tensor([context], dtype=torch.float32).to(device)
            
            # Get prediction
            output = model(context_tensor)
            next_token = output[0, -1, :].cpu().numpy()
            
            # Apply temperature sampling
            if temperature > 0:
                # Add noise for diversity
                noise = np.random.normal(0, temperature * 0.1, next_token.shape)
                next_token = next_token + noise
            
            # Clamp values to valid ranges
            next_token[0] = np.clip(next_token[0], 0, 1)  # pitch
            next_token[1] = np.clip(next_token[1], 0, 1)  # velocity
            next_token[2] = np.clip(next_token[2], 0, 1)  # duration
            next_token[3] = np.clip(next_token[3], 0, 1)  # delta_time
            
            generated = torch.cat((generated ,torch.from_numpy(next_token).unsqueeze(0).to(device)))
            
            # Stop if we generate an end token
            if np.allclose(next_token, [1.0, 1.0, 1.0, 1.0], atol=0.1):
                break
    
    return generated

def tokens_to_midi(tokens, output_file='generated_music.mid'):
    """Convert tokens back to MIDI file"""
    # Create new MIDI file
    midi = miditoolkit.MidiFile()
    
    # Create piano instrument - fix for newer miditoolkit versions
    try:
        # Try newer API first
        piano = miditoolkit.Instrument(program=1, is_drum=False, name='Piano')
    except AttributeError:
        # Fallback to older API
        piano = miditoolkit.containers.Instrument(program=1, is_drum=False, name='Piano')
    
    current_time = 0
    
    for token in tokens:
        # Skip special tokens
        if (token == [0.0, 0.0, 0.0, 0.0] or 
            token == [1.0, 1.0, 1.0, 1.0] or 
            token == [-1.0, -1.0, -1.0, -1.0]):
            continue
        
        # Convert back to MIDI values with proper denormalization
        pitch = int(token[0] * 127)
        velocity = int(token[1] * 127)
        
        # Convert duration back from log scale
        duration_normalized = token[2]
        if duration_normalized > 0:
            duration_log = duration_normalized * (np.log(4.0) - np.log(0.1)) + np.log(0.1)
            duration = np.exp(duration_log)
        else:
            duration = 0.1  # minimum duration
        
        # Convert delta_time back from log scale
        delta_normalized = token[3]
        if delta_normalized > 0:
            delta_log = delta_normalized * (np.log(2.0) - np.log(0.01)) + np.log(0.01)
            delta_time = np.exp(delta_log)
        else:
            delta_time = 0.01  # minimum delta
        
        # Update current time
        current_time += delta_time
        
        # Create note - fix for newer miditoolkit versions
        try:
            # Try newer API first
            note = miditoolkit.Note(
                pitch=max(21, min(108, pitch)),  # Piano range
                velocity=max(1, min(127, velocity)),
                start=int(current_time * 480),  # Convert to ticks
                end=int((current_time + duration) * 480)
            )
        except AttributeError:
            # Fallback to older API
            note = miditoolkit.containers.Note(
                pitch=max(21, min(108, pitch)),  # Piano range
                velocity=max(1, min(127, velocity)),
                start=int(current_time * 480),  # Convert to ticks
                end=int((current_time + duration) * 480)
            )
        
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.dump(output_file)
    print(f"Generated MIDI saved as {output_file}")

# Load best model
#model.load_state_dict(torch.load('best_music_transformer.pth', map_location=device))
#print("Loaded best model for generation")
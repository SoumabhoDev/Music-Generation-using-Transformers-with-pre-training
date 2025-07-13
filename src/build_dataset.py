import numpy as np
from tqdm import tqdm
from src.midi_loader import is_piano

def create_training_data(midi_files, context_length=32, max_notes_per_song=100000000):
    """Create training data with better preprocessing"""
    X, y = [], []
    
    # Special tokens
    START_TOKEN = [0.0, 0.0, 0.0, 0.0]  # Start of sequence
    END_TOKEN = [1.0, 1.0, 1.0, 1.0]    # End of sequence
    PAD_TOKEN = [-1.0, -1.0, -1.0, -1.0]  # Padding
    
    for song_idx, song in enumerate(tqdm(midi_files, desc="Processing songs")):
        # Get piano tracks
        piano_tracks = [inst for inst in song.instruments if is_piano(inst)]
        if not piano_tracks:
            continue
        
        # Collect all piano notes
        notes = []
        for piano in piano_tracks:
            notes.extend(piano.notes)
        
        if len(notes) == 0:
            continue
        
        # Sort notes by start time
        notes = sorted(notes, key=lambda x: x.start)
        
        # Limit notes per song to prevent memory issues
        if len(notes) > max_notes_per_song:
            notes = notes[:max_notes_per_song]
        
        # Convert notes to tokens
        tokens = [START_TOKEN]  # Start with special token
        
        for i, note in enumerate(notes):
            # Normalize values to [0, 1] range
            pitch = note.pitch / 127.0  # MIDI pitch range 0-127
            velocity = note.velocity / 127.0  # MIDI velocity range 0-127
            
            # Duration in ticks, convert to seconds and normalize
            duration_ticks = note.end - note.start
            duration_seconds = duration_ticks / 480.0  # 480 ticks per quarter note
            # Use log scale for better distribution, clamp between 0.1 and 4.0 seconds
            duration_seconds = max(0.1, min(duration_seconds, 4.0))
            # Normalize using log scale to prevent extreme values
            duration = (np.log(duration_seconds) - np.log(0.1)) / (np.log(4.0) - np.log(0.1))
            
            # Time delta from previous note in ticks
            if i > 0:
                delta_ticks = note.start - notes[i-1].start
                delta_seconds = max(0.01, delta_ticks / 480.0)  # At least 10ms
                delta_seconds = min(delta_seconds, 2.0)  # Max 2 seconds
                # Use log scale for delta time as well
                delta_time = (np.log(delta_seconds) - np.log(0.01)) / (np.log(2.0) - np.log(0.01))
            else:
                delta_time = 0.0
            
            token = [pitch, velocity, duration, delta_time]
            tokens.append(token)
        
        tokens.append(END_TOKEN)  # End with special token
        
        # Create training sequences
        for i in range(len(tokens) - context_length):
            context = tokens[i:i + context_length]
            target = tokens[i + context_length]
            
            X.append(context)
            y.append(target)
    
    return X, y
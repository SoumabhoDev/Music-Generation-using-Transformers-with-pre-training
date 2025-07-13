import glob
import os
from tqdm import tqdm
import miditoolkit

def is_piano(instrument):
    """Check if instrument is a piano"""
    return not instrument.is_drum and 0 <= instrument.program <= 7

def quantize_time(t, res=50):  # 50ms steps for good resolution
    """Quantize time to fixed resolution"""
    return int(round(t / res) * res)

def load_midi_files(directory, max_files=10000000000):
    """Load MIDI files from directory"""
    midi_files = glob.glob(os.path.join(directory, '**', '*.mid'), recursive=True)
    midi_files += glob.glob(os.path.join(directory, '**', '*.midi'), recursive=True)
    
    # Limit number of files for testing
    midi_files = midi_files[:max_files]
    
    midi_objects = []
    for file_path in tqdm(midi_files, desc="Loading MIDI files"):
        try:
            midi_objects.append(miditoolkit.MidiFile(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return midi_objects
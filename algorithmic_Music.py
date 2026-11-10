import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100

def midi_to_freq_432(midi_note: int) -> float:
    """Calculates frequency based on A4 = 432 Hz equal temperament."""
    return 432.0 * (2.0 ** ((midi_note - 69) / 12.0))

def generate_tone(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesizes a musical tone with harmonics and an ADSR amplitude envelope."""
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Additive synthesis: Fundamental + soft harmonics for a warm sound
    waveform = (
        1.0 * np.sin(2 * np.pi * freq * t) +
        0.4 * np.sin(2 * np.pi * (freq * 2) * t) +
        0.15 * np.sin(2 * np.pi * (freq * 3) * t)
    )
    
    # Basic ADSR envelope
    total_samples = len(t)
    attack = int(sr * min(0.05, duration * 0.1))
    decay = int(sr * min(0.08, duration * 0.15))
    release = int(sr * min(0.12, duration * 0.25))
    sustain = total_samples - (attack + decay + release)
    
    env = np.concatenate([
        np.linspace(0, 1, attack),
        np.linspace(1, 0.7, decay),
        np.full(max(0, sustain), 0.7),
        np.linspace(0.7, 0, release)
    ])[:total_samples]
    
    return waveform * env

def generate_algorithmic_track(num_steps: int = 48) -> np.ndarray:
    """
    Generates music using an algorithmic Markov-like walk 
    over the A Minor Pentatonic scale centered around A4 (432 Hz).
    """
    # MIDI notes for A Minor Pentatonic: A3 to C6
    # 69 is A4 = 432 Hz
    scale = [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84]
    
    # Available note durations in seconds (e.g., sixteenth, eighth, quarter notes)
    durations = [0.25, 0.5, 0.75]
    
    current_index = scale.index(69)  # Start at A4 (432 Hz)
    audio_buffer = []
    
    print(f"Generating {num_steps} algorithmic notes tuned to 432 Hz...")
    
    for _ in range(num_steps):
        # Step rule: Tend to make small interval steps, occasionally leap
        step = np.random.choice([-2, -1, 0, 1, 2, 3], p=[0.15, 0.3, 0.1, 0.3, 0.1, 0.05])
        current_index = max(0, min(len(scale) - 1, current_index + step))
        
        midi_note = scale[current_index]
        freq = midi_to_freq_432(midi_note)
        dur = float(np.random.choice(durations, p=[0.4, 0.45, 0.15]))
        
        # 10% chance of a rest (silence)
        if np.random.rand() < 0.10:
            audio_buffer.append(np.zeros(int(SAMPLE_RATE * dur)))
        else:
            note_audio = generate_tone(freq, dur)
            audio_buffer.append(note_audio)
            
    track = np.concatenate(audio_buffer)
    
    # Normalize to 16-bit range to prevent clipping
    track = track / np.max(np.abs(track)) * 0.8
    return (track * 32767).astype(np.int16)

if __name__ == "__main__":
    audio_data = generate_algorithmic_track()
    output_filename = "algorithmic_music_432hz.wav"
    wavfile.write(output_filename, SAMPLE_RATE, audio_data)
    print(f"Saved: {output_filename}")

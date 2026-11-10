import numpy as np
from scipy.io import wavfile

# --- Audio Configuration ---
SAMPLE_RATE = 48000           # 48 kHz
TOTAL_DURATION = 600.0        # 600 seconds (10 minutes)
TOTAL_SAMPLES = int(SAMPLE_RATE * TOTAL_DURATION)

def midi_to_freq_432(midi_note: int) -> float:
    """Calculates frequency based on A4 = 432 Hz equal temperament."""
    return 432.0 * (2.0 ** ((midi_note - 69) / 12.0))

def add_to_buffer(buffer: np.ndarray, audio: np.ndarray, start_sample: int, pan: float = 0.5):
    """
    Mixes a mono audio signal into a stereo buffer at a given sample index,
    applying constant-power stereo panning (pan: 0.0 = Left, 1.0 = Right).
    """
    end_sample = min(start_sample + len(audio), len(buffer))
    slice_len = end_sample - start_sample
    if slice_len <= 0:
        return

    audio_slice = audio[:slice_len]
    left_gain = np.cos(pan * (np.pi / 2.0))
    right_gain = np.sin(pan * (np.pi / 2.0))

    buffer[start_sample:end_sample, 0] += audio_slice * left_gain
    buffer[start_sample:end_sample, 1] += audio_slice * right_gain

def synthesize_pad(freq: float, duration: float) -> np.ndarray:
    """Generates a soft, detuned warm pad with smooth cosine fade-ins and fade-outs."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Detuned oscillators for subtle analog chorus warmth
    wave = (
        0.50 * np.sin(2 * np.pi * freq * t) +
        0.25 * np.sin(2 * np.pi * (freq * 1.002) * t) +
        0.25 * np.sin(2 * np.pi * (freq * 0.998) * t) +
        0.10 * np.sin(2 * np.pi * (freq * 2.0) * t)
    )
    
    # Smooth raised-cosine attack and release to avoid clicks
    fade_len = int(SAMPLE_RATE * min(duration * 0.35, 3.5))
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
    
    env = np.ones_like(t)
    env[:fade_len] = fade_in
    env[-fade_len:] = fade_out
    
    return wave * env

def synthesize_pluck(freq: float, duration: float) -> np.ndarray:
    """Generates a bell/chime sound with harmonics and exponential decay."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Exponential decay envelope
    decay_rate = 3.5 / duration
    env = np.exp(-decay_rate * t)
    
    # Micro-attack (5ms) to eliminate initial click
    attack_len = int(SAMPLE_RATE * 0.005)
    env[:attack_len] *= np.linspace(0, 1, attack_len)
    
    # Overtone profile (fundamental + octave + minor third overtone)
    wave = (
        0.70 * np.sin(2 * np.pi * freq * t) +
        0.20 * np.sin(2 * np.pi * (freq * 2.0) * t) +
        0.10 * np.sin(2 * np.pi * (freq * 2.99) * t)
    )
    return wave * env

def generate_ambient_piece():
    print(f"Allocating 10-minute stereo buffer ({SAMPLE_RATE} Hz)...")
    # Pre-allocate 600s stereo float buffer (~230 MB RAM)
    master_buffer = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)

    # ---------------------------------------------------------
    # Layer 1: Evolving Pad Chords (D-centered meditative modal)
    # ---------------------------------------------------------
    print("Generating Layer 1: Warm 432 Hz Drone Pads...")
    chords = [
        [50, 57, 62],       # D minor (D3, A3, D4)
        [50, 58, 62],       # Bb/D (D3, Bb3, D4)
        [50, 55, 60],       # C/D sus (D3, G3, C4)
        [50, 57, 60, 64]    # Dm7/9 (D3, A3, C4, E4)
    ]
    
    pad_time = 0.0
    while pad_time < TOTAL_DURATION:
        chord = chords[np.random.randint(len(chords))]
        chord_dur = float(np.random.uniform(9.0, 14.0))
        
        for note in chord:
            freq = midi_to_freq_432(note)
            pad_audio = synthesize_pad(freq, chord_dur) * 0.18
            start_idx = int(pad_time * SAMPLE_RATE)
            add_to_buffer(master_buffer, pad_audio, start_idx, pan=np.random.uniform(0.3, 0.7))
            
        # Overlap chords by 3 seconds for seamless crossfading
        pad_time += chord_dur - 3.0

    # ---------------------------------------------------------
    # Layer 2: Algorithmic Pluck Melody (Random Walk / Bells)
    # ---------------------------------------------------------
    print("Generating Layer 2: Algorithmic Chimes & Plucks...")
    # D Minor Pentatonic Scale (centered around 432 Hz A4 = 69)
    scale = [57, 60, 62, 65, 67, 69, 72, 74, 77, 79, 81, 84]
    
    current_idx = scale.index(69)  # Start at A4 (432 Hz)
    pluck_time = 2.0  # Start shortly after pads swell in

    while pluck_time < (TOTAL_DURATION - 5.0):
        # Markov-style step: small steps are favored over large leaps
        step = np.random.choice([-3, -2, -1, 0, 1, 2, 3], p=[0.05, 0.15, 0.25, 0.1, 0.25, 0.15, 0.05])
        current_idx = int(np.clip(current_idx + step, 0, len(scale) - 1))
        
        note = scale[current_idx]
        freq = midi_to_freq_432(note)
        dur = float(np.random.uniform(1.5, 4.0))
        
        # Random velocity & stereo placement
        velocity = float(np.random.uniform(0.12, 0.35))
        pan = float(np.random.uniform(0.15, 0.85))
        
        pluck_audio = synthesize_pluck(freq, dur) * velocity
        start_idx = int(pluck_time * SAMPLE_RATE)
        add_to_buffer(master_buffer, pluck_audio, start_idx, pan=pan)
        
        # Delay before next note (semi-randomized rhythmic spacing)
        pluck_time += float(np.random.choice([0.75, 1.25, 1.75, 2.5], p=[0.3, 0.4, 0.2, 0.1]))

    # ---------------------------------------------------------
    # Master Processing: Fade-in, Fade-out & 16-bit Normalization
    # ---------------------------------------------------------
    print("Applying master envelope and normalising audio...")
    
    # 4-second master fade-in, 8-second master fade-out
    fade_in_len = int(SAMPLE_RATE * 4.0)
    fade_out_len = int(SAMPLE_RATE * 8.0)
    
    master_buffer[:fade_in_len, :] *= np.linspace(0, 1, fade_in_len)[:, None]
    master_buffer[-fade_out_len:, :] *= np.linspace(1, 0, fade_out_len)[:, None]

    # Normalize to -1 dB peak (0.89) to prevent digital clipping
    peak = np.max(np.abs(master_buffer))
    if peak > 0:
        master_buffer = (master_buffer / peak) * 0.89

    return (master_buffer * 32767).astype(np.int16)

if __name__ == "__main__":
    audio = generate_ambient_piece()
    filename = "ambient_432hz_10min_48k.wav"
    
    print(f"Writing to {filename}...")
    wavfile.write(filename, SAMPLE_RATE, audio)
    print("Done! Enjoy your 10-minute 432 Hz soundscape.")

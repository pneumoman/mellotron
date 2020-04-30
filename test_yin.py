from yin import compute_yin
from utils import load_wav_to_torch
import numpy as np

def get_f0(audio, sampling_rate=22050, frame_length=1024,
           hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
    f0, harmonic_rates, argmins, times = compute_yin(audio, sampling_rate,
                                                     frame_length,
                                                     hop_length,
                                                     f0_min,
                                                     f0_max,
                                                     harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad
    f0 = np.array(f0, dtype=np.float32)
    return f0

filename1 = 'untitled.wav'

sampling_rate = fs = 44100
max_wav_value=32768.0
filter_length=win_length=frameSize = 1024//2
hop_length=hopSize = filter_length//4
n_mel_channels=80
mel_fmin=0.0
mel_fmax=8000.0
f0_min=80
f0_max=880
harm_thresh=0.25

audio, sampling_rate = load_wav_to_torch(filename1)
f0 = get_f0(audio.cpu().numpy(), sampling_rate,
                 filter_length, hop_length, f0_min,
                 f0_max, harm_thresh)
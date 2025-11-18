import soundfile as sf
import librosa
import numpy as np

noisy_path = "noisy_wavs/noisy2.wav"
NFFT = 1024
win_length = 1024
hop_length = 512
pre_noise_frames = 50
out_path = "output_wavs/spectral_substract.wav"

noisy_data, sr = librosa.load(noisy_path, sr=None)
x = librosa.stft(noisy_data, n_fft=NFFT, win_length=win_length, hop_length=hop_length, window='hann')
magnitude, phase = librosa.magphase(x)
mag = magnitude.copy()

# noise estimate
mag_noise = np.mean(magnitude[:,:pre_noise_frames], axis=1)

# substract
bins, frames = magnitude.shape
for i in range(frames):
    mag[:, i] = (magnitude[:, i] > mag_noise) * (magnitude[:, i] - mag_noise)

out = librosa.istft(mag * phase, n_fft=NFFT, win_length=win_length, hop_length=hop_length, window='hann')
sf.write(out_path, out, sr)

# diff
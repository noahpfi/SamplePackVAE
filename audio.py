import librosa
import librosa.display
import librosa.feature
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def to_mel_spectrogram(filename, params):
    audio, sr = librosa.load(filename, sr=params['SAMPLE_RATE'], duration=params['SAMPLE_LENGTH'])
    spec = librosa.feature.melspectrogram(y=audio, sr=sr,
                                          n_fft=params['N_FFT'],
                                          hop_length=params['HOP_LENGTH'],
                                          n_mels=params['MEL_BANDS'])
    spec = pad_mel_spectrogram(spec, params)
    return spec, sr


def pad_mel_spectrogram(spec, params):
    if spec.shape[1] < params['PAD_LENGTH']:
        spec = np.pad(spec, ((0, 0), (0, params['PAD_LENGTH'] - spec.shape[1])))
    return spec


def remove_dims_mel_spectrogram(spec):
    return np.squeeze(spec, axis=(0, 3))


def expand_dims_mel_spectrogram(spec):
    return np.expand_dims(np.expand_dims(spec, axis=0), axis=3)


def save_mel_spectrogram(spec, filename, params):
    librosa.display.specshow(spec, sr=params['SAMPLE_RATE'], cmap='magma')
    plt.savefig(filename, bbox_inches='tight')


def save_mel_spectrogram_as_wave(spec, filename, params):
    wav = librosa.feature.inverse.mel_to_audio(spec, sr=params['SAMPLE_RATE'],
                                               n_fft=params['N_FFT'],
                                               hop_length=params['HOP_LENGTH'])
    sf.write(filename, wav, params['SAMPLE_RATE'])

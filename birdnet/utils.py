import numpy as np
import librosa

def _normalize(audio):
    """
    Normalize the audio array.

    Args:
        audio (numpy.ndarray): Input audio array.

    Returns:
        numpy.ndarray: Normalized audio array.
    """
    min_val, max_val = np.min(audio), np.max(audio)
    normalized = (audio - min_val) / (max_val - min_val)
    return normalized

def read_audio(file, sr=4800):
    """
    Read audio file and return the audio data along with the sample rate.

    Args:
        file (str): Path to the audio file.
        sr (int): Desired sampling rate.

    Returns:
        numpy.ndarray: Audio data.
        int: Sample rate.
    """
    audio, sr = librosa.load(file, sr=sr)
    return audio, sr

def audio_to_spectrogram(audio, tmin=0, tmax=100):
    """
    Convert audio signal into Mel spectrogram.

    Args:
        audio (numpy.ndarray): Input audio signal.
        tmin (float): Minimum time in seconds.
        tmax (float): Maximum time in seconds.

    Returns:
        numpy.ndarray: Normalized Mel spectrogram.
    """
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio[int(4800 * tmin):int(4800 * tmax)], sr=4800)
    # Convert to decibel scale
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Apply normalization
    normalized = _normalize(mel_spectrogram_db)
    return normalized

def visualize(signals, labels, Y, row, col):
    fig, axes = plt.subplots(row, col, figsize=(9, 5))
    # Plot three gibbon presence events
    for i, indices in enumerate(labels):
      for j, idx in enumerate(indices):
          # generate the spectrogram
          spectrogram = audio_to_spectrogram(signals[idx])
          # plot the generated spectrogram
          # axes.colorbar(format='%+2.0f dB')
          librosa.display.specshow(spectrogram, x_axis='time', sr=4800, hop_length=256, ax=axes[i, j])
          axes[i, j].set_title(f"{Y[idx]}")
          axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()
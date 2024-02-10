import math
import numpy as np
import pandas as pd
from scipy import signal
import librosa
from tabulate import tabulate  # assuming you want to use tabulate for printing tables
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from scipy import signal
import librosa

class BaseEvaluation(ABC):
    """
    Base class for evaluation pipelines.
    """

    def __init__(self, model, path, audio_to_spectrogram):
        """
        Initialize the EvaluationPipeline.

        Args:
            model: A trained model.
            path (str): Path to audio files.
            audio_to_spectrogram: Function to convert audio to spectrogram.
        """
        self.model = model
        self.path = path
        self.audio_to_spectrogram = audio_to_spectrogram
        self.preds = []

    @abstractmethod
    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        """
        Design a lowpass Butterworth filter.

        Args:
            cutoff (float): Cutoff frequency of the filter.
            nyq_freq (float): Nyquist frequency.
            order (int): Order of the filter.

        Returns:
            tuple: Numerator (b) and denominator (a) of the filter.
        """
        pass

    @abstractmethod
    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        """
        Apply a lowpass Butterworth filter to the data.

        Args:
            data (numpy.ndarray): Input data.
            cutoff_freq (float): Cutoff frequency of the filter.
            nyq_freq (float): Nyquist frequency.
            order (int): Order of the filter.

        Returns:
            numpy.ndarray: Filtered data.
        """
        pass

    @abstractmethod
    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        """
        Downsample the audio file.

        Args:
            amplitudes (numpy.ndarray): Audio data.
            original_sr (int): Original sample rate.
            new_sample_rate (int): New sample rate.

        Returns:
            numpy.ndarray: Downsampled audio data.
        """
        pass

    @abstractmethod
    def predict_on_entire_file(self, audio, sample_rate, lowpass_cutoff, downsample_rate, nyquist_rate):
        """
        Predict presence/absence of target sound in the entire audio file.

        Args:
            audio (numpy.ndarray): Input audio data.
            sample_rate (int): Sample rate of the audio.
            lowpass_cutoff (float): Cutoff frequency for lowpass filter.
            downsample_rate (int): Rate for downsampling.
            nyquist_rate (float): Nyquist frequency.

        Returns:
            list: List of predictions for each segment.
        """
        pass

    @abstractmethod
    def evaluate_files(self, lowpass_cutoff, downsample_rate, nyquist_rate):
        """
        Evaluate multiple audio files.

        Args:
            lowpass_cutoff (float): Cutoff frequency for lowpass filter.
            downsample_rate (int): Rate for downsampling.
            nyquist_rate (float): Nyquist frequency.
        """
        pass

    @abstractmethod
    def create_annotation(self, i, low_freq=1200, high_freq=2000):
        """
        Create annotation CSV for predicted segments.

        Args:
            i (int): Index of the file.
            low_freq (int): Low frequency for annotation.
            high_freq (int): High frequency for annotation.
        """
        pass

    @abstractmethod
    def get_indices(self, i):
        """
        Get indices of predicted segments.

        Args:
            i (int): Index of the file.

        Returns:
            numpy.ndarray: Indices of predicted segments.
        """
        pass

    @abstractmethod
    def group_consecutives(self, i, step=1):
        """
        Group consecutive indices into segments.

        Args:
            i (int): Index of the file.
            step (int): Step size for grouping.

        Returns:
            list: List of grouped indices.
        """
        pass

class Evaluation(BaseEvaluation):
    """
    A pipeline for evaluating audio files using a trained model.
    """
    def __init__(self, model, path, audio_to_spectrogram):
        """
        Initialize the EvaluationPipeline.

        Args:
            model: A trained model.
            path (str): Path to audio files.
            audio_to_spectrogram: Function to convert audio to spectrogram.
        """
        self.model = model
        self.path = path
        self.audio_to_spectrogram = audio_to_spectrogram
        self.preds = []

    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        """
        Design a lowpass Butterworth filter.

        Args:
            cutoff (float): Cutoff frequency of the filter.
            nyq_freq (float): Nyquist frequency.
            order (int): Order of the filter.

        Returns:
            tuple: Numerator (b) and denominator (a) of the filter.
        """
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        """
        Apply a lowpass Butterworth filter to the data.

        Args:
            data (numpy.ndarray): Input data.
            cutoff_freq (float): Cutoff frequency of the filter.
            nyq_freq (float): Nyquist frequency.
            order (int): Order of the filter.

        Returns:
            numpy.ndarray: Filtered data.
        """
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        """
        Downsample the audio file.

        Args:
            amplitudes (numpy.ndarray): Audio data.
            original_sr (int): Original sample rate.
            new_sample_rate (int): New sample rate.

        Returns:
            numpy.ndarray: Downsampled audio data.
        """
        return librosa.resample(y=amplitudes, orig_sr=original_sr, target_sr=new_sample_rate, res_type='kaiser_fast')

    def predict_on_entire_file(self, audio, sample_rate, lowpass_cutoff, downsample_rate, nyquist_rate):
        """
        Predict presence/absence of target sound in the entire audio file.

        Args:
            audio (numpy.ndarray): Input audio data.
            sample_rate (int): Sample rate of the audio.
            lowpass_cutoff (float): Cutoff frequency for lowpass filter.
            downsample_rate (int): Rate for downsampling.
            nyquist_rate (float): Nyquist frequency.

        Returns:
            list: List of predictions for each segment.
        """
        filtered = self.butter_lowpass_filter(audio, lowpass_cutoff, nyquist_rate)
        amplitudes = self.downsample_file(filtered, sample_rate, downsample_rate)
        file_duration = len(amplitudes) / sample_rate
        segments = math.floor(file_duration) - 4
        predictions = []

        for position in range(0, segments):
            start_position = position
            end_position = start_position + 4
            audio_segment = amplitudes[start_position * downsample_rate:end_position * downsample_rate]
            S = self.audio_to_spectrogram(audio_segment)
            S = np.reshape(S, (1, 128, 76, 1))
            S = np.repeat(S, 3, axis=-1)
            softmax = self.model.predict(S, verbose=0)
            binary_prediction = np.argmax(softmax, -1)
            predictions.append('absence' if binary_prediction[0] == 0 else 'presence')

        return predictions

    def evaluate_files(self, lowpass_cutoff, downsample_rate, nyquist_rate):
        """
        Evaluate multiple audio files.

        Args:
            lowpass_cutoff (float): Cutoff frequency for lowpass filter.
            downsample_rate (int): Rate for downsampling.
            nyquist_rate (float): Nyquist frequency.
        """
        files = [f"{self.path}check{i}.wav" for i in range(1, 11)]
        table = []

        for i, file in enumerate(files):
            testfile, sr = librosa.load(file)
            predictions = self.predict_on_entire_file(testfile, sr, lowpass_cutoff, downsample_rate, nyquist_rate)
            self.preds.append(predictions)
            count1 = predictions.count('presence')
            count2 = predictions.count('absence')
            total_segments = len(predictions)
            table.append([f'Test {i}', count1, count2, total_segments])

        headers = ["File", "Gabbon (0) Count", "Non-Gabbon (1) Count", "Total Segments"]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def create_annotation(self, i, low_freq=1200, high_freq=2000):
        """
        Create annotation CSV for predicted segments.

        Args:
            i (int): Index of the file.
            low_freq (int): Low frequency for annotation.
            high_freq (int): High frequency for annotation.
        """
        start_time = []
        end_time = []
        groups = self.group_consecutives(i)
        for group in groups:
            start_time.append(group[0])
            end_time.append(group[-1] + 4)

        df_preds = pd.DataFrame({
            'start(sec)': start_time,
            'end(sec)': end_time,
            'low(freq)': low_freq,
            'high(freq)': high_freq,
            'label': 'predicted'
        })
        df_preds.to_csv("annotation.csv", index=False)

    def get_indices(self, i):
        """
        Get indices of predicted segments.

        Args:
            i (int): Index of the file.

        Returns:
            numpy.ndarray: Indices of predicted segments.
        """
        df = pd.DataFrame(self.preds[i], columns=['BinaryPrediction'])
        indices = df[df['BinaryPrediction'] == 'presence'].index.values
        return indices

    def group_consecutives(self, i, step=1):
        """
        Group consecutive indices into segments.

        Args:
            i (int): Index of the file.
            step (int): Step size for grouping.

        Returns:
            list: List of grouped indices.
        """
        run = []
        result = [run]
        expect = None
        indices = self.get_indices(i)
        for v in indices:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

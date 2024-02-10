import numpy as np
from abc import ABC, abstractmethod
from utils import *
from dataclasses import dataclass
from preprocessing import *



@dataclass
class DataParameters:
    """
    A class that defines a set of parameters for data processing and analysis.
    """
    positive_class: list[str] = ['gibbon']
    negative_class: list[str] = ['no-gibbon']
    lowpass_cutoff: int = 2000
    downsample_rate: int = 4800
    nyquist_rate: int = 2400
    segment_duration: int = 4
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: int = 4000
    f_max: int = 9000
    species_folder: str = '.'
    file_type: str = 'svl'
    audio_extension: str = '.wav'



class DatasetBase(ABC):
    def __init__(self, X, y):
        """
        Initialize the dataset with features and labels.

        Args:
            X (array-like): Features.
            y (array-like): Labels.
        """
        self.X = X
        self.y = y
        self.new_presence = []
        self.new_absence  = []

    @abstractmethod
    def preprocess(self, new_example=True, quantity=[100, 200]):
        """
        Preprocess the data including new example creation, augmentation, and normalization.

        Args:
            new_example (bool): Whether to generate new examples.
            quantity (list): A list containing the quantity of new examples to generate for presence and absence.

        Returns:
            numpy.ndarray: Augmented features.
            numpy.ndarray: Labels.
        """
        pass

    @abstractmethod
    def create_train_test_split(self, data, test_size=0.0, random_state=42):
        """
        Split the data into training and testing sets.

        Args:
            data (numpy.ndarray): Features.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.

        Returns:
            numpy.ndarray: Training features.
            numpy.ndarray: Testing features.
            numpy.ndarray: Training labels.
            numpy.ndarray: Testing labels.
        """
        pass

    @abstractmethod
    def randomly_select_spectrogram(self, label):
        """
        Randomly select a spectrogram based on the label.

        Args:
            label (str): The label of the spectrogram to select.

        Returns:
            numpy.ndarray: Selected spectrogram.
            int: Index of the selected spectrogram.
        """
        pass

    @abstractmethod
    def generate_new_spectrograms(self, quantity, presence=True):
        """
        Generate new spectrograms for augmentation.

        Args:
            quantity (list): A list containing the quantity of new examples to generate for presence and absence.
            presence (bool): Whether to generate examples with presence or absence.

        Returns:
            numpy.ndarray: New spectrograms.
            numpy.ndarray: New labels.
        """
        pass


class Dataset(DatasetBase):
    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.new_presence = []
        self.new_absence = []
    
    def extract(self, path):
        with zipfile.ZipFile(path, 'r') as zip_file:
            zip_file.extractall()
    def create_dataset(self):
        pre_pro = Preprocessing(species_folder, lowpass_cutoff,
                downsample_rate, nyquist_rate,
                segment_duration,
                positive_class, negative_class,n_fft,
                hop_length, n_mels, f_min, f_max, file_type,
                audio_extension)

        self.X, self.y = pre_pro.create_dataset(False)

        # We save the pickle X and y variables to disk so that we don't have to
        # pre-process the data everytime we want to train a model
        pre_pro.save_data_to_pickle(X, Y)

    def _augment_clip(self, signal, minm=10, maxm=40):
        """
        Clipping Distortion: Augmentation

        The percentage of points that will be clipped is drawn from a uniform distribution between
        the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
        30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
        """
        random_percent = np.random.uniform(minm, maxm)
        # Calculate the threshold values for clipping
        clip_min = np.percentile(signal, (100 - random_percent) / 2)
        clip_max = np.percentile(signal, (100 + random_percent) / 2)
        return np.clip(signal, clip_min, clip_max)

    def _polarity_inversion(self, audio):
        """
        Polarity Inversion: Augmentation

        The function to reverse all the the data upside down

        Return the negative of the original data. Its importance is reflected on the above sections
        """
        return -audio
    def preprocess(self, new_exmple=True, quantity=[100, 200]):
        """
        Function to preprocess the data

        This include new example creation, augmentation and normilization
        """
        # If genearate the new example is "True" and append them to original data
        if new_exmple:
          # when this function called it will combine the original
          # data with the new examples
          self.generate_new_spectrograms(quantity)
          # generate the data with the absence
          self.generate_new_spectrograms(quantity, presence=False)
        # apply augmentation on each of the datapoint
        augmented = []
        for audio in self.X:
            # convert the audio and normilize
            S_dB = audio_to_spectrogram(audio)
            # apply clip distortion augmnetation
            clip   = self._augment_clip(S_dB)
            # apply polarity inversion augmnetation
            invert = self._polarity_inversion(S_dB)
            # add a dimension for the tensorflow model (tensorflow expects channel dimension)
            S_dB, clip, invert  = S_dB[:, :, np.newaxis], clip[:, :, np.newaxis], invert[:, :, np.newaxis]
            # concatenate original and the other two augmneted datapoint into the last dimension
            _augmnent = np.concatenate((S_dB, clip, invert), axis=-1)
            # collected the augmented data to the list
            augmented.append(_augmnent)
        return np.array(augmented), self.y

    def create_train_test_split(self, data, test_size=0.0, random_state=42):
        """
        Function to split the data into training and testing sets.
        """
        # Create the label encode instances
        call_order = ['no-gibbon', 'gibbon']
        # Converting categorical string labels ('gibbons' and 'no-gibbon) to 0s and 1s
        for index, call_type in enumerate(call_order):
            self.y = np.where(self.y == call_type, index, self.y)
        # convert the label to categorical
        label = to_categorical(self.y, num_classes=2)
        # split the data in to train and test
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=random_state, shuffle=True)
        return X_train, X_test, y_train, y_test

    def randomly_select_spectrogram(self, label):
        if label == 'gibbon':
            indices = np.where(self.y == 'gibbon')[0]
        else:
            indices = np.where(self.y == 'no-gibbon')[0]
        # randomly generate the index
        idx = np.random.randint(0, len(indices))
        return self.X[indices[idx]], idx

    def generate_new_spectrograms(self, quantity, presence=True):
        # quantity: [ presence, ansence ]
        # _ indicates internal newly generated datas
        _spectrogram, _targets = [], []

        if presence:
          for i in range(quantity[0]):
            # select spec randomly
            selected_spectrogram, idx = self.randomly_select_spectrogram('gibbon')
            _spectrogram.append(selected_spectrogram)
            _targets.append('gibbon')
          # append it to the class presence acumulator variable
          self.new_presence.append(_spectrogram)
        else:
          for i in range(quantity[1]):
            # select spec randomly
            selected_spectrogram, idx = self.randomly_select_spectrogram('no-gibbon')
            _spectrogram.append(selected_spectrogram)
            _targets.append('no-gibbon')
          # append it to the clas absence acummulator variable
          self.new_absence.append(_spectrogram)
        # Concatenate the newly formed data
        self.X = np.concatenate((self.X, _spectrogram), axis=0)
        self.y = np.concatenate((self.y, np.array(_targets)), axis=0)
        return _spectrogram, np.array(_targets)

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Base Trainer class providing an interface for model training.
    """

    def __init__(self, model):
        """
        Initialize the BaseTrainer with a given model.

        Args:
            model: A compiled model (e.g., Keras model).
        """
        self.model = model
        self.history = None

    @abstractmethod
    def train(self, X_train, y_train, epochs=5, validation_split=0.2, batch_size=32):
        """
        Abstract method to train the model.

        Args:
            X_train: Input training data.
            y_train: Target training data.
            epochs (int): Number of epochs for training.
            validation_split (float): Fraction of training data to use as validation data.
            batch_size (int): Size of mini-batches for training.
        """
        pass

    @abstractmethod
    def plot_training_history(self):
        """
        Abstract method to plot the training and validation history of the model.
        """
        pass

class Trainer(BaseTrainer):
    """
    A Training Pipeline class for training and visualizing deep learning models.
    """
    def __init__(self, model):
        """
        Initialize the TrainerPipeline with a given model.

        Args:
            model: A compiled Keras model.
        """
        self.model = model
        self.history = None

    def train(self, X_train, y_train, epochs=5, validation_split=0.2, batch_size=32):
        """
        Train the model.

        Args:
            X_train: Input training data.
            y_train: Target training data.
            epochs (int): Number of epochs for training.
            validation_split (float): Fraction of training data to use as validation data.
            batch_size (int): Size of mini-batches for training.
        """
        self.history = self.model.fit(X_train, y_train, epochs=epochs,
                                      validation_split=validation_split, batch_size=batch_size)

    def plot_training_history(self):
        """
        Plot the training and validation history of the model.
        """
        if self.history is None:
            raise RuntimeError("Model has not been trained yet. Call 'train' method first.")

        metrics = ['accuracy', 'loss']
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for ax, metric in zip(axes, metrics):
            ax.plot(self.history.history[metric], label='Train')
            ax.plot(self.history.history['val_' + metric], label='Validation')
            ax.set_title('Model ' + metric.capitalize())
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel('Epoch')
            ax.legend()

        plt.tight_layout()
        plt.show()

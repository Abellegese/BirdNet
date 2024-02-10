from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class Mel2DConvModel:
    def __init__(self, input_shape):
        """
        Initialize the Mel2DConvModel.

        Args:
            input_shape (tuple): Input shape of the model.
        """
        self._input_shape = input_shape
        self._model = self._build_model()

    def _build_model(self):
        """
        Build the convolutional neural network model.

        Returns:
            keras.models.Sequential: Built model.
        """
        model = Sequential([
            Conv2D(filters=16, kernel_size=2, input_shape=self._input_shape, activation='relu'),
            MaxPool2D(pool_size=2),
            Conv2D(filters=32, kernel_size=2, activation='relu'),
            MaxPool2D(pool_size=2),
            Conv2D(filters=64, kernel_size=2, activation='relu'),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(units=30, activation='relu'),
            Dense(2, activation='softmax')
        ])
        return model

    def compile_model(self):
        """
        Compile the model.

        Returns:
            None
        """
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        """
        Print the summary of the model.

        Returns:
            None
        """
        self._model.summary()

    @property
    def input_shape(self):
        """
        Get the input shape of the model.

        Returns:
            tuple: Input shape of the model.
        """
        return self._input_shape

    @property
    def model(self):
        """
        Get the built model.

        Returns:
            keras.models.Sequential: Built model.
        """
        return self._model

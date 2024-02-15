from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Add, GlobalAveragePooling2D
from keras.applications import EfficientNetB0

class Mel2DConvModel:
    def __init__(self, input_shape, model_type='basic'):
        """
        Initialize the Mel2DConvModel.

        Args:
            input_shape (tuple): Input shape of the model.
            model_type (str): Type of model to build. Options: 'basic', 'resnet10', 'efficientnet'.
        """
        self._input_shape = input_shape
        self._model_type = model_type
        self._model = self._build_model()

    def _build_model(self):
        """
        Build the convolutional neural network model.

        Returns:
            keras.models.Sequential: Built model.
        """
        if self._model_type == 'basic':
            model = self._build_basic_model()
        elif self._model_type == 'resnet10':
            model = self._build_resnet10()
        elif self._model_type == 'efficientnet':
            model = self._build_efficientnet()
        else:
            raise ValueError("Invalid model type. Please choose from 'basic', 'resnet10', or 'efficientnet'.")
        
        return model

    def _build_basic_model(self):
        model = Sequential([
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self._input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(30, activation='relu'),
            Dense(2, activation='softmax')
        ])
        return model

    def _build_resnet10(self):
        def residual_block(x, filters, strides=(1, 1)):
            shortcut = x
            x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
            x = Activation('relu')(x)
            x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
            if strides != (1, 1) or shortcut.shape[-1] != filters:
                shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            return x

        inputs = Input(shape=self._input_shape)
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        for _ in range(2):
            x = residual_block(x, 64)

        x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        for _ in range(2):
            x = residual_block(x, 128)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        for _ in range(2):
            x = residual_block(x, 256)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _build_efficientnet(self):
        model = EfficientNetB0(input_shape=self._input_shape, weights=None, include_top=True, classes=2)
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

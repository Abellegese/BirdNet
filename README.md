<img src="https://happymag.tv/wp-content/uploads/2020/04/Webp.net-resizeimage-9-2.jpg" height=300px; width=800px>

# BirdNet: Bird Voice Classifier

[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

## Introduction

This project is a bird classifier built using deep learning techniques. The classifier is trained on audio recordings of bird sounds to predict the species of birds present in the recordings. It leverages the power of convolutional neural networks (CNNs) to automatically learn features from the audio spectrograms and make accurate predictions.

Bird classification is a challenging task due to the diverse range of bird species and the variability in their vocalizations. Traditional methods of bird classification rely on manual feature engineering and classification algorithms, which can be time-consuming and may not generalize well to unseen data.

In this project, we use deep learning techniques to automate the process of feature extraction and classification. Convolutional neural networks (CNNs) have shown remarkable success in various audio classification tasks, including speech recognition and music classification. We apply similar techniques to the task of bird classification, leveraging the spectral patterns present in bird vocalizations.


## Model
<img src="vgg16_xml.jpg" height=300px>

## Getting Started

## Usage

### Command-Line Interface (CLI)

To train your model using the command-line interface, you can use the `train.py` script with the following options:

```bash
python train.py --epochs 10 --batch_size 64 --validation_split 0.1 --new_example True --quantity 100 200
```

- `--epochs`: Number of epochs for training (default: 5).
- `--batch_size`: Batch size for training (default: 32).
- `--validation_split`: Validation split ratio (default: 0.2).
- `--new_example`: Whether to generate new examples (default: True).
- `--quantity`: Quantity of new examples to generate (default: [100, 200]). But the choice is fully upto the users. If you need balanced dataset put the quantity that can make both the positive and negative example equal.

### Python API

You can also use the Python API to train your model programmatically. 

```python
from birdnet.train import Trainer
from birdnet.dataset import Dataset

# Initialize Trainer Pipeline
trainer = Trainer()

# Load data using Dataset class
dataset = Dataset(X, y)  # Replace X and y with your actual data

# Preprocess data
X_train, y_train = dataset.preprocess(new_example=True, quantity=[100, 200])

# Train the model using TrainerPipeline
trainer.train_model(X_train, y_train, epochs=10, validation_split=0.1, batch_size=64)
```

#### Evaluating Audio Files

To evaluate audio files using the trained model, you can use the `predict.py` script with the following options:

```bash
python predict.py --model_path "path/to/saved_model" --audio_file "path/to/audio_file.wav" --output_file "predictions.csv"
```

- `--model_path`: Path to the saved TensorFlow model file.
- `--audio_file`: Path to the audio file to be evaluated.
- `--output_file`: Path to save the predictions CSV file.

### Dataset
The dataset consists of audio recordings collected from Intaka Island, Cape Town, South Africa, for training a convolutional neural network (CNN)-based bird classifier. Recordings were gathered by placing audio recorders in various habitats across the island, for almost 5 hours.  In total, around 4 additional hours of recordings were obtained. Most recordings used a sampling rate of 48,000 Hz to capture sound details.

Link: https://doi.org/10.5281/zenodo.10659537


### visualize

Visualizes spectrograms of audio signals with their corresponding labels.

## Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from birdnet.utils import audio_to_spectrogram, visualize

# Load audio data
# Y: The training data
# label: the index of the positive and negative class [[pos1, pos2],[neg1, neg2]]
# row and col: the size of the figure
spectrogram = audio_to_spectrogram(audio)
visualize(signals, labels, Y, row, col)
```
### Report 
The report paper can be found https://github.com/Abellegese/BirdNet/blob/main/example/abellegese_DL4E_Assignment_2.pdf

## Acknowledgments

- Parts of the code in this repository were adapted from [Emmanuel Dufourq's GitHub repository](https://github.com/emmanueldufourq).

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


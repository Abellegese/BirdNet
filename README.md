<img src="https://happymag.tv/wp-content/uploads/2020/04/Webp.net-resizeimage-9-2.jpg" height=300px; width=800px>

# BirdNet: Bird Voice Classifier

[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

## Introduction

BirdNet is a library that provides a data preprocessing, model development, training and analysis tools. It aims to provide a state of the art bird classifier models for the community.

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
- `--quantity`: Quantity of new examples to generate (default: [100, 200]).

### Python API

You can also use the Python API to train your model programmatically. Here's an example:

```python
from audio_evaluation_pipeline import TrainerPipeline, Dataset

# Initialize TrainerPipeline
trainer = TrainerPipeline()

# Load data using Dataset class
dataset = Dataset(X, y)  # Replace X and y with your actual data

# Preprocess data
X_train, y_train = dataset.preprocess(new_example=True, quantity=[100, 200])

# Train the model using TrainerPipeline
trainer.train_model(X_train, y_train, epochs=10, validation_split=0.1, batch_size=64)
```

Replace `X` and `y` with your actual data. You can adjust the parameters (`epochs`, `batch_size`, `validation_split`, `new_example`, `quantity`) as needed.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Make sure to replace placeholders like `your-username`, `your-repo`, `train.py`, `audio-evaluation-pipeline`, `X`, `y`, etc., with actual values specific to your project.

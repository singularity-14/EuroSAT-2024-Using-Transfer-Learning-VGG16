# Land Use and Land Cover Classification using Transfer Learning with VGG16

## Overview

In this project, Transfer Learning with the VGG16 model is employed for Land Use and Land Cover (LULC) classification. Using the EuroSAT dataset, the model achieves high accuracy in identifying 10 different classes.

## Dataset

The EuroSAT dataset consists of Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 different LULC classes. The dataset is split into training and validation sets.

## Model Architecture

The VGG16 model pre-trained on the ImageNet dataset is utilized as the base model. The fully connected layers of VGG16 are modified to suit the LULC classification task. The model is then compiled with appropriate loss and optimization functions.

## Data Preparation

- Images are preprocessed and augmented using the `ImageDataGenerator` class in TensorFlow.
- The dataset is divided into training and validation sets.

## Training

- Convolutional layers of the VGG16 model are frozen to retain pre-trained weights.
- The model is trained on the EuroSAT dataset for a specified number of epochs.
- Training and validation accuracy and loss are monitored during the training process.

## Visualization

- Training and validation accuracy over epochs are visualized using line plots.
- Training and validation loss over epochs are also plotted for analysis.

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV

## Usage

1. **Dataset Preparation**:
   - Ensure the EuroSAT dataset is downloaded or accessible.
   - Organize the dataset into appropriate directories for training and validation.

2. **Model Training**:
   - Run the provided Python script `land_use_classification.py`.
   - Ensure the correct paths to the dataset and required dependencies are specified.
   - Adjust hyperparameters as needed.

3. **Evaluation**:
   - Evaluate the model performance based on training and validation accuracy and loss plots.

## Customization

- Experiment with different pre-trained models and architectures.
- Fine-tune hyperparameters such as learning rate, dropout rate, and batch size.
- Explore advanced data augmentation techniques for improving model robustness.

## File Structure

- `lulc.ipynb`: Main Python script for model training and evaluation.
- `README.md`: This file providing an overview and instructions.
- `data/`: Directory containing the EuroSAT dataset split into training and validation sets.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

- Rachit Patel


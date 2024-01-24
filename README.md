# MNIST Digit Recognition using Convolutional Neural Network (CNN)

## Overview

This repository contains code for a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras for the task of digit recognition on the MNIST dataset. The model achieves high accuracy and has been trained and evaluated on the MNIST dataset, a benchmark dataset for handwritten digit recognition.

## Requirements

To run the code in this repository, you need to have the following dependencies installed:

```bash
pip install numpy matplotlib keras
```

Make sure you have the TensorFlow library installed. If not, you can install it using:

```bash
pip install tensorflow
```

## Dataset

The MNIST dataset is automatically downloaded and loaded using TensorFlow's `mnist.load_data()` function. It consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels.

## Data Pre-processing

The loaded images are pre-processed as follows:

- Normalized to the range [0, 1].
- Reshaped to include a channel dimension (for compatibility with CNN architecture).
- One-hot encoded the labels.

## Model Architecture

The CNN model architecture consists of the following layers:

1. Convolutional layer with 32 filters, kernel size (3, 3), and ReLU activation.
2. MaxPooling layer with pool size (2, 2).
3. Dropout layer with a dropout rate of 0.25.
4. Convolutional layer with 64 filters, kernel size (3, 3), and ReLU activation.
5. MaxPooling layer with pool size (2, 2).
6. Dropout layer with a dropout rate of 0.25.
7. Flatten layer to transform the 2D feature maps to a 1D vector.
8. Fully connected (Dense) layer with 128 units and ReLU activation.
9. Fully connected (Dense) layer with 10 units and softmax activation.

## Model Compilation and Training

The model is compiled using categorical cross-entropy loss and the Adam optimizer. It is then trained for 10 epochs on the training data with a batch size of 32. Validation data is used during training to monitor model performance on unseen data.

## Model Evaluation

The trained model is evaluated on the test set, and both the loss and accuracy are printed.

## Visualization

The README provides a visual representation of the training history, plotting both training and testing accuracy over the epochs.

## Testing on a Single Image

A single test sample is selected, and the trained model predicts the digit. The actual and predicted digits, along with the image, are displayed.

Feel free to use and modify the code according to your needs. Happy coding!

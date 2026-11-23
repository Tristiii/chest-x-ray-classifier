# Chest X-ray Classifier
A convolutional neural network (CNN) that recognizes pathology on chest x-rays.

## 1. Introduction

X-rays are a great way to localize pathology of the chest, in particular of the heart and lungs. The low cost, low exposure to harmful radioactivity and the ability to quickly utilize it, makes it a great tool to use to get a general overview of the chest.

Furthermore, machine learning is becoming more prominent in the medical field to make decisions in important settings, such as in the emergency room (ER) or the intensive care unit (ICU).

In this repository, multiple convolutional neural networks (CNNs) were tested to investigate whether chest pathology could reliably be recognized by a machine learning model.

## 2. Methods

### 2.1 Dataset

The dataset that was used is ChestMNIST, part of MedMNIST, found in the [following link](https://medmnist.com/). This is a database which contains MNIST-like (28x28 pixels) images to use for classification training in the medical field.

Distribution of the data (total 112.120 samples):
- Training: 78.468 samples.
- Validation: 11.219 samples.
- Test: 22.433 samples.

### 2.2 Model

For now, only a variant of DenseNet121 has been implemented.

## 3. Results

The DenseNet121-like model obtains an F1-score of 0,17.
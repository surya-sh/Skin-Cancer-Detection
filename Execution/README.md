---
tags:
- skin-cancer-detection
- medical
- image-classification
- tensorflow
- keras
license: mit
datasets:
- HAM10000
metrics:
- accuracy
model-index:
  - name: Skin Cancer Detection Model
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: HAM10000
          type: HAM10000
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.73
---

# Skin Cancer Detection Model

This model is built to detect different types of skin cancer from dermatoscopic images. It was trained using the **HAM10000** dataset and is designed to classify seven types of skin lesions. The goal of this model is to assist in the early detection of skin cancer, particularly in regions where temperatures are rising, making skin cancer detection increasingly important.

## Model Overview
The model is a Convolutional Neural Network (CNN) developed using **TensorFlow** and **Keras**. It takes input images of skin lesions and predicts one of the seven skin cancer classes. The model achieved an accuracy of **73%**, which can be further improved with fine-tuning and larger datasets.

## Classes Detected
The model predicts the following seven types of skin cancer:

1. **akiec**: Actinic Keratoses and Intraepithelial Carcinoma (pre-cancerous)
2. **bcc**: Basal Cell Carcinoma (a common type of skin cancer)
3. **bkl**: Benign Keratosis (non-cancerous lesion)
4. **df**: Dermatofibroma (benign skin lesion)
5. **nv**: Melanocytic Nevus (a common mole)
6. **vasc**: Vascular Lesions (non-cancerous lesions of blood vessels)
7. **mel**: Melanoma (most dangerous type of skin cancer)

### Explanation of Each Class:
- **Actinic Keratoses (akiec)**: A rough, scaly patch on the skin that develops from years of sun exposure. It is considered pre-cancerous and can lead to squamous cell carcinoma.
- **Basal Cell Carcinoma (bcc)**: A type of skin cancer that starts in the basal cells and is often caused by UV radiation. It rarely spreads to other parts of the body.
- **Benign Keratosis (bkl)**: Non-cancerous skin growths that resemble moles but do not pose any risk.
- **Dermatofibroma (df)**: A common, benign fibrous skin lesion, usually found on the legs.
- **Melanocytic Nevus (nv)**: Also known as a mole, these are benign proliferations of melanocytes but can sometimes develop into melanoma.
- **Vascular Lesions (vasc)**: Benign growths formed by abnormal blood vessels, generally harmless.
- **Melanoma (mel)**: The most serious type of skin cancer, arising from melanocytes. Early detection is critical for effective treatment.

## Dataset
The **HAM10000** dataset, which contains 10,015 dermatoscopic images of different skin lesions, was used to train this model. The dataset is publicly available and widely used for research on skin lesion classification.

## Model Architecture
The model is based on a Convolutional Neural Network (CNN) architecture with multiple layers of convolution, max-pooling, and fully connected layers to classify images. The input images were resized to **224x224** pixels and normalized to ensure uniformity across the dataset.

Key architecture components include:
- Convolution layers with ReLU activation
- Max Pooling layers for down-sampling
- Fully connected dense layers
- Softmax activation for the final classification

## Training
The model was trained on 80% of the HAM10000 dataset, with 20% used for validation. **Data augmentation** was applied to the training set to improve generalization, including techniques like rotation, flipping, and scaling.

The training configuration:
- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**
- Batch size: **32**
- Epochs: **10**

## Metrics
- **Accuracy**: The primary evaluation metric was accuracy. The model achieved **73%** accuracy on the validation set.
  
  Further improvements can be achieved with additional training or fine-tuning using transfer learning methods.

## Execution Steps

### Operating System

In this project we are assuming you will be using an Unix based operating system which might include the following:- 

    1. MacOS
    2. Any distro of Linux
    3. Any variant of BSD
    4. WSL in Windows 11

### Install Package Manager
Homebrew is an os agnostic package manager which we will be using to install the required software and libraries.
Homebrew webpage: https://brew.sh/
Installation:-
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Software/Libraries
We will now use homebrew to install all the required software:-
```bash
brew install git python3
```
We will now use pip to install the required libraries
```bash
pip install numpy tensorflow gradio
```
### Getting the Source Code
The source code is available in github which could be obtained via git
```bash
git clone https://github.com/surya-sh/Skin-Cancer-Detection.git
```

### Running the Code
We would need source the packages and run the code
```bash
cd Execution/
source bin/activate
python3 app.py
```

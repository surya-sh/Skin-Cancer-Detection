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

## How to Use
You can use this model to classify skin lesion images. Below is an example code snippet to load the model and make predictions:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model from Hugging Face
model = load_model('skin_cancer_model.h5')

# Preprocess the image and make a prediction
img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Mapping class index to class name
class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
print(f"Predicted class: {class_names[predicted_class]}")
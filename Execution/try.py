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


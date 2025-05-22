
from transformers import pipeline

# Load the model from the Hugging Face Hub
classifier = pipeline("image-classification", model="VRJBro/skin-cancer-detection")

# Example usage
image_path = "test_image.jpg"
result = classifier(image_path)
print(result)

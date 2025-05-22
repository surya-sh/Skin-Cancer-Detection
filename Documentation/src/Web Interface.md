---
updated: 2025-04-23T11:43
---
#### <center> APPENDICES <br><br> APPENDIX 1 <br><br> SAMPLE CODING</center>


#### Web Interface

```python
import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

model = load_model("skin_cancer_model.h5")

def webview(image):
    img = tf.image.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
    file_path = f"{class_names[predicted_class]}.md"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
        return md_content
    except FileNotFoundError:
        return f"Markdown file '{file_path}' not found."
# Create a Gradio interface
iface = gr.Interface(
    fn=webview,
    # A slider is used to ensure the input is an integer between 1 and 5.
    inputs="image",
    outputs="markdown",
    title="Skin Cancer Detection",
    description="Enter the image of affected area"
)
# Launch the interface
iface.launch()
```
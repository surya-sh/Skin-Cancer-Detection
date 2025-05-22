
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Average, Input

# Loading the individual models.
model1 = load_model('EfficientNetV2M.h5')
model2 = load_model('EfficientNetV2S.h5')
model3 = load_model('InceptionResNetV2.h5')
model4 = load_model('XceptionNet.h5')

# Checking that the input shapes of the models match. 
input_shape = model1.input_shape[1:]  

# Creating a shared input layer
input_layer = Input(shape=input_shape)

# Passing the input through each model
output1 = model1(input_layer)
output2 = model2(input_layer)
output3 = model3(input_layer)
output4 = model4(input_layer)

# Combining the outputs using an averaging layer.
combined_output = Average()([output1, output2, output3, output4])

# Creating the ensemble model.
ensemble_model = Model(inputs=input_layer, outputs=combined_output)

# Compiling the ensemble model (necessary to further train it)
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Saving the ensemble model.
ensemble_model.save('skin_cancer_model.h5')

print("Done")

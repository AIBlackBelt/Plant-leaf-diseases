import os

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

classes = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# load image and preprocess it in order to be fed to the model
def load_img(path):
  img = cv2.imread(path,cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  resized = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
  scaled = resized/255
  batch_img = np.reshape(scaled,(1,224,224,3))
  return batch_img

# Recreate the exact same model, including its weights and the optimizer
inceptionv3_model = tf.keras.models.load_model('inceptionv3.h5')

# Show the model architecture
inceptionv3_model.summary()

image_path = input("submit the path where the image is located :\n")

preprocessed_image = load_img(image_path)

# predict the label of the image that was read
y_pred = inceptionv3_model(preprocessed_image)

numpy_prediction_array = y_pred[0].numpy()

# get index of the highest probability computed among the whole distribution of 38 labels
prediction = np.argmax(numpy_prediction_array)

# add a dimension to embedd the image in a batch
image_to_display = np.reshape(preprocessed_image,(224,224,3))

# display image along with label
plt.imshow(image_to_display)
plt.title(classes[prediction])
plt.show()


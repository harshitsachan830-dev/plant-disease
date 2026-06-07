import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/plant_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

with open("models/class_names.json") as f:
    class_names = json.load(f)

img_path = input("Image path: ")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\nPrediction:", class_names[class_index])
print("Confidence:", round(confidence, 2), "%")
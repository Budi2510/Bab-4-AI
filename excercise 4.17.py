import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

model = VGG16(weights="imagenet")

img_path = "dodo.png"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Ganti n menjadi 2, 3, atau 4 untuk melihat layer berbeda.
n = 2

layer_outputs = [model.layers[n].output]
activation_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=layer_outputs
)

activations = activation_model.predict(img_array, verbose=0)
feature_maps = activations[0]

plt.figure(figsize=(12,12))

for i in range(min(16, feature_maps.shape[-1])):
    plt.subplot(4,4,i+1)
    plt.imshow(feature_maps[:,:,i], cmap="viridis")
    plt.axis("off")

plt.tight_layout()
plt.show()

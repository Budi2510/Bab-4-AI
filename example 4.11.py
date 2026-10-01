# Example 4.11 - VGG16 image classification
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = VGG16(weights='imagenet')
img_path = 'DODO.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
predictions = model.predict(x, verbose=0)
results = decode_predictions(predictions)
print(results)

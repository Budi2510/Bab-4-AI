import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load model VGG19
model = VGG19(weights="imagenet")

# Lokasi gambar
img_path = "DODO.png"

# Load gambar
img = image.load_img(img_path, target_size=(224, 224))

# Ubah gambar menjadi array
x = image.img_to_array(img)

# Tambahkan dimensi batch
x = np.expand_dims(x, axis=0)

# Preprocessing untuk VGG19
x = preprocess_input(x)

# Prediksi
predictions = model.predict(x, verbose=0)

# Ambil 5 prediksi teratas
results = decode_predictions(predictions, top=5)

print("Hasil prediksi:")

for result in results[0]:
    print(result)
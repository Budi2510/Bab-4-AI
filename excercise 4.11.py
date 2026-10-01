import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# Ukuran input VGG19
IMAGE_SIZE = 224

# Load model VGG19
print("Memuat model VGG19...")
model = VGG19(weights="imagenet")
print("Model berhasil dimuat!")

# Membuka webcam
camera = cv2.VideoCapture(0)

# Cek apakah webcam berhasil dibuka
if not camera.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

while True:

    # Mengambil frame dari webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Error: Gagal membaca frame webcam.")
        break

    # Membalik gambar agar seperti mirror
    cam_frame = cv2.flip(cam_frame, 1)

    # -----------------------------
    # PREPROCESSING UNTUK VGG19
    # -----------------------------

    # Resize hanya untuk proses prediksi
    frame = cv2.resize(cam_frame, (IMAGE_SIZE, IMAGE_SIZE))

    # Ubah gambar menjadi array
    numpy_image = img_to_array(frame)

    # Tambahkan dimensi batch
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Preprocessing sesuai VGG19
    processed_image = preprocess_input(image_batch)

    # -----------------------------
    # PREDIKSI
    # -----------------------------

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Ambil 1 prediksi terbaik
    label = decode_predictions(
        predictions,
        top=1
    )

    # Nama objek
    name = label[0][0][1]

    # Confidence
    score = label[0][0][2] * 100

    # Format teks
    text = "{} {:.1f}%".format(
        name,
        score
    )

    # -----------------------------
    # TAMPILKAN HASIL
    # -----------------------------

    cv2.putText(
        cam_frame,
        "VGG19: " + text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Tampilkan webcam
    cv2.imshow(
        "VGG19 Webcam",
        cam_frame
    )

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Tutup webcam
camera.release()
cv2.destroyAllWindows()
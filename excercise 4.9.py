from tensorflow.keras.applications import VGG19

model = VGG19(weights="imagenet")
model.summary()

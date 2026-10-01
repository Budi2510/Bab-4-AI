from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    DenseNet201
)

choice = input(
    "Pilih model (vgg16/resnet50/mobilenetv2/densenet201): "
).lower()

models = {
    "vgg16": VGG16,
    "resnet50": ResNet50,
    "mobilenetv2": MobileNetV2,
    "densenet201": DenseNet201
}

if choice not in models:
    print("Model tidak tersedia.")
else:
    model = models[choice](weights="imagenet")
    model.summary()

from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    DenseNet201,
    InceptionV3
)

models = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "DenseNet201": DenseNet201,
    "InceptionV3": InceptionV3
}

for name, model_class in models.items():
    print("\nLoading:", name)
    model = model_class(weights="imagenet")
    print("Parameters:", model.count_params())

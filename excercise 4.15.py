import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

model = VGG16(weights="imagenet")

n = 2
layer = model.layers[n]

if len(layer.get_weights()) == 0:
    print("Layer", layer.name, "tidak memiliki filter/bobot.")
else:
    filters, biases = layer.get_weights()

    print("Layer:", layer.name)
    print("Filter shape:", filters.shape)
    print("Bias shape:", biases.shape)

    n_filters = min(4, filters.shape[3])

    plt.figure(figsize=(8,2))
    for i in range(n_filters):
        f = filters[:,:,:,i]
        plt.subplot(1, n_filters, i+1)
        plt.imshow(f[:,:,0], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

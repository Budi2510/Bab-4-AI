import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

model = VGG16(weights="imagenet")

n = 2
layer = model.layers[n]

if len(layer.get_weights()) == 0:
    print("Layer tidak memiliki filter.")
else:
    filters, biases = layer.get_weights()

    n_filters = filters.shape[3]
    cols = int(np.ceil(np.sqrt(n_filters)))
    rows = int(np.ceil(n_filters / cols))

    plt.figure(figsize=(12,12))

    for i in range(n_filters):
        f = filters[:,:,:,i]
        plt.subplot(rows, cols, i+1)
        plt.imshow(f[:,:,0], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

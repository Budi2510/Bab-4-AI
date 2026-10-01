# Example 4.17 - CNN Visualize Filters (Parts 1-6)
# Based on the Chapter 4 program for visualizing VGG19 filters and feature maps.
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Part 1: load and summarize VGG19
model = VGG19()
model.summary()

# Part 2: display layer numbers and names
for n, layer in enumerate(model.layers):
    print(n, layer.name)

# Part 3: display convolutional filter and bias shapes
for n, layer in enumerate(model.layers):
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)

# Part 4: visualize the first four filters of convolutional layer 1
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print('Color channels:', s[2])
print('Filter size:', s[0], s[1])
print('Total number of filters:', s[3])
f_min, f_max = filters.min(), filters.max()
if f_max != f_min:
    filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 4, 1
plt.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[2]):
        ax = plt.subplot(n_filters, s[2], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
plt.show()

# Part 5: helper for plotting feature maps
def plot_feature_maps(feature_maps):
    col = 8
    row = int(np.ceil(feature_maps.shape[3] / col))
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            if ix > feature_maps.shape[3]:
                break
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

# Part 6: display feature maps of layer 1 when Elephant.jpg is available.
try:
    feature_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    img = load_img('Elephant.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature_maps = feature_model.predict(img, verbose=0)
    print('Feature maps:', feature_maps.shape)
    plot_feature_maps(feature_maps)
except FileNotFoundError:
    print("Elephant.jpg not found; Parts 1-5 can still be run.")

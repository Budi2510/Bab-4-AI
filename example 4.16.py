# Example 4.16 - AutoEncoder for MNIST
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape)
print(x_test.shape)

def build_autoencoder(img_shape, code_size):
    encoder = Sequential([
        InputLayer(shape=img_shape), 
        Flatten(), 
        Dense(code_size)
    ])
    
    # Membungkus np.prod(img_shape) dengan int() untuk menghindari error tipe data pada Dense layer
    decoder = Sequential([
        InputLayer(shape=(code_size,)), 
        Dense(int(np.prod(img_shape)), activation='relu'), 
        Reshape(img_shape)
    ])
    
    return encoder, decoder

IMG_SHAPE = x_train[0].shape
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)
inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
print(autoencoder.summary())

history = autoencoder.fit(x=x_train, y=x_train, epochs=20, validation_data=(x_test, x_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def show_image(x):
    plt.imshow(np.clip(x, 0, 1), cmap='gray')

def visualize(img, encoder, decoder):
    code = encoder.predict(img[None], verbose=0)[0]
    reco = decoder.predict(code[None], verbose=0)[0]
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.title('Original'); show_image(img)
    plt.subplot(1, 3, 2); plt.title('Code'); plt.imshow(code.reshape([code.shape[-1] // 2, -1]), cmap='gray')
    plt.subplot(1, 3, 3); plt.title('Reconstructed'); show_image(reco)
    plt.show()

for i in range(5):
    visualize(x_test[i], encoder, decoder)
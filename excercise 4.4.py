from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(64, (5,5), activation="relu"),
    MaxPooling2D(),
    Conv2D(64, (5,5), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(120, activation="relu"),
    Dense(84, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()

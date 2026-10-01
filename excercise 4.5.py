from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

model = Sequential([
    Input(shape=(224,224,3)),
    Flatten(),
    Dense(2048, activation="relu"),
    Dense(2048, activation="relu"),
    Dense(1000, activation="softmax")
])

model.summary()

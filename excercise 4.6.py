from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (5,5), activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

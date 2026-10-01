import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(5, activation="relu"),
    Dense(5, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mse"]
)

start = time.time()

model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=0
)

training_time = time.time() - start
results = model.evaluate(X_test, y_test, verbose=0)

print("Training time:", training_time, "seconds")
print("Loss:", results[0])
print("MSE:", results[1])

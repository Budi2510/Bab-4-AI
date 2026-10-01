# Example 4.3 - Multiple Layer Perceptron Classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7)

model = Sequential([
    Dense(10, activation='relu', input_shape=[X_train.shape[1]]),
    Dense(10, activation='relu'),
    Dense(1)
])
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.fit(X_train, y_train, batch_size=50, validation_split=0.2,
          epochs=100, verbose=1)
results = model.evaluate(X_test, y_test, verbose=0)
print('loss:', results[0])
print('mse:', results[1])

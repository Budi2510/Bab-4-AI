from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input

def show_model(filters, kernel):
    model = Sequential([
        Input(shape=(28,28,1)),
        Conv2D(filters, kernel, activation="relu")
    ])
    print(f"\nFilters={filters}, Kernel={kernel}")
    model.summary()

show_model(32, (5,5))
show_model(64, (5,5))
show_model(64, (3,3))

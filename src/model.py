# Install once in terminal:
# pip install tensorflow

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Preprocess
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# Model
model = Sequential([
    Embedding(10000, 128),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3, batch_size=128)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", acc)
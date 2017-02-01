import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot

# Generate data
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000, 1))
labels = to_categorical(labels, 10)

# Model
model = Sequential()

model.add(Dense(output_dim=32, input_dim=784))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
training_history = model.fit(data, labels, nb_epoch=5, batch_size=32)

# Test
data = np.random.random((100, 784))
labels = np.random.randint(10, size=(100, 1))
labels = to_categorical(labels, 10)
loss_and_metrics = model.evaluate(data, labels, batch_size=32)

print('\n' + '-' * 40)
print("Metrics:")
print(loss_and_metrics)

print('-' * 40)
print(training_history) # This object contains everything, including the calculated weights & bias

# Plot the model topology
plot(model, to_file='model.png', show_shapes=True)
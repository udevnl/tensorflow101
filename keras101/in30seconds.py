import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

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
# This will show the loss (=the cost) and the accuracy (percentage correct)
training_history = model.fit(data, labels, nb_epoch=30, verbose=2, batch_size=32)

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
# plot(model, to_file='model.png', show_shapes=True)

# Even cooler stuff - See https://keras.io/models/about-keras-models/
# ------------------------------------------------------------------------------------------------------------------

# Print summary
model.summary()

# Print model topology (can be used to build model from!)
config = model.get_config()
print('\n' + 40 * '-' + '\nSerialized model definition:')
print(config)
# model = Model.from_config(config)
# or, for Sequential:
# model = Sequential.from_config(config)

# Print the weights of the bias of the first layer
all_weights = model.get_weights()
print('\n' + 40 * '-' + '\nBias values of layer 1:')
print(all_weights[1])

# Now plot it!
layer1_weights = np.transpose(all_weights[0])

for x in range(32):
    plt.subplot(6, 6, x + 1)
    plt.axis('off')
    plt.imshow(layer1_weights[x].reshape((28, 28)))

plt.show()
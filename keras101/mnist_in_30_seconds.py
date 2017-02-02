import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.examples.tutorials.mnist import input_data

# Get MMIST data
mnist = input_data.read_data_sets('./mnist_data', one_hot=True)

# Model
model = Sequential()

model.add(Dense(output_dim=32, input_dim=784))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
training_history = model.fit(
    mnist.train.images,
    mnist.train.labels,
    nb_epoch=50,
    verbose=2,
    batch_size=100
)

# Test
loss_and_metrics = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=32)
print("\n\nTest results:")
print(loss_and_metrics)

# Now plot it!
all_weights = model.get_weights()

layer1_weights = np.transpose(all_weights[0])

for x in range(32):
    plt.subplot(6, 6, x + 1)
    plt.axis('off')
    plt.imshow(layer1_weights[x].reshape((28, 28)), cmap='plasma')

plt.show()
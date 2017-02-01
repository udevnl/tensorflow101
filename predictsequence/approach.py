import tensorflow as tf
import numpy as np
from random import shuffle

# Data sets - the data set is a continues list of values
# Where training to predict the next value based on the previous 'feature_count' samples

# Generate data
feature_count = 4
sample_count = 10
train_count = 7
all_data = [4.5 + 4.5 * np.cos(x / (5.0 + x / 100)) for x in range(feature_count + sample_count)]
# all_data = [x%10 for x in range(feature_count + sample_count)]

sample_features = np.array([[np.float32(all_data[n + m]) for n in range(feature_count)] for m in range(sample_count)])
sample_labels = np.array([ np.int(0.5 + all_data[x + feature_count]) for x in range(sample_count)])

# Split into train and test data
random_order = [x for x in range(sample_count)]
shuffle(random_order)

train_features = sample_features[random_order[:train_count]]
train_labels = sample_labels[random_order[:train_count]]

test_features = sample_features[random_order[train_count:]]
test_labels = sample_labels[random_order[train_count:]]

# print(random_order)
#
# print(sample_features)
# print(sample_labels)
# print(len(all_data))
#
print(train_features)
print(train_labels)

# Build the network

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=feature_count)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[feature_count, feature_count*2, feature_count],
                                            n_classes=10,
                                            model_dir="/tmp/approach5")

# Fit model.
classifier.fit(x=train_features,
               y=train_labels,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_features,
                                     y=test_labels)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[1.0, 2.0, 3.0, 4.0],
     [8.0, 7.0, 6.0, 5.0],
     ], dtype=np.float32)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))

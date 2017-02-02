from PyQt4 import QtCore

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


def map_motion_curve_to_feature(motion_curve, sample_count):
    return motion_curve.flatten().reshape((1, sample_count * 2))


class Predictor(QtCore.QObject):
    """
        Online learning predictor
    """

    def __init__(self, button_count, curve_sample_count):
        super().__init__()
        self.button_count = button_count
        self.curve_sample_count = curve_sample_count
        self.model = self.__create_model(button_count, curve_sample_count)

    def __create_model(self, button_count, curve_sample_count):
        feature_count = curve_sample_count * 2

        model = Sequential()
        model.add(Dense(output_dim=32, input_dim=feature_count))
        model.add(Activation("relu"))
        model.add(Dense(output_dim=button_count))
        model.add(Activation("softmax"))

        # The optimizer
        sgd = SGD(lr=0.1)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, motion_curve, pressed_button_index):
        print('Training')
        x = map_motion_curve_to_feature(motion_curve, self.curve_sample_count)
        y = to_categorical(np.array([pressed_button_index]), self.button_count)
        self.model.train_on_batch(x, y)

    def predict(self, motion_curve):
        motion_curve = map_motion_curve_to_feature(motion_curve, self.curve_sample_count)
        x = motion_curve
        r = self.model.predict(x).flatten()
        print(r)
        maxIndex = np.argmax(r)
        maxValue = r[maxIndex]

        if maxValue < 0.5:
            print('Too uncertain.')
        else:
            print("{0:.2f} sure of {1:1d}".format(maxValue, maxIndex))
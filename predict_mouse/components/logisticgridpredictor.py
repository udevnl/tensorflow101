import numpy as np
from PyQt4 import QtCore
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD

from predict_mouse.components.shared import Events


class LogisticGridPredictor(QtCore.QObject):
    """
        Online learning predictor
    """

    def __init__(self, x_size, y_size, max_xp, max_yp, sigma, curve_sample_count):
        super().__init__()

        # Input parameters
        self.sigma = sigma
        self.max_yp = max_yp
        self.max_xp = max_xp
        self.y_size = y_size
        self.x_size = x_size
        self.curve_sample_count = curve_sample_count

        # Derived parameters
        self.feature_count = curve_sample_count * 2
        self.output_count = x_size * y_size

        # Create a grid of coordinates
        self.grid = np.array([[x, y] for x in np.linspace(-1, 1, x_size) for y in np.linspace(-1, 1, y_size)])

        self.model = self.__create_model()
        self.train_data = np.empty((0, self.feature_count), np.float32)
        self.train_labels = np.empty((0, self.output_count), np.float32)

    def __create_model(self):

        model = Sequential()
        model.add(Dense(output_dim=64, input_dim=self.feature_count))
        model.add(Activation("relu"))
        model.add(Dense(output_dim=32))
        model.add(Activation("relu"))
        model.add(Dense(output_dim=self.output_count))
        model.add(Activation("softmax"))

        # The optimizer
        sgd = SGD(lr=0.01)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def __map_motion_curve_to_feature(self, motion_curve):
        motion_curve = motion_curve - motion_curve[0]
        return motion_curve.flatten()

    def __map_curve_to_final_grid_position(self, motion_curve):
        normalized_final_position = (motion_curve[-1] - motion_curve[0]) / [self.max_xp, self.max_yp]

        # Calculate the similarity between the grid positions and the normalized_final_position
        # (this uses a gaussian curve)
        return np.exp(
            -np.linalg.norm(self.grid - normalized_final_position, axis=1) / (2 * np.square(self.sigma))
        ).flatten()

    def set_train_data(self, motion_curves):
        # Filter out the curves that do not have enough data
        good_curves = [curve for curve in motion_curves if len(curve) > self.curve_sample_count]

        m = len(good_curves)
        print("Loaded {} out of {} curves.".format(m, len(motion_curves)))

        x_curves = [self.__map_motion_curve_to_feature(curve[0:self.curve_sample_count]) for curve in good_curves]
        y_labels = [self.__map_curve_to_final_grid_position(curve) for curve in good_curves]

        self.train_data = np.array(x_curves).reshape((m, self.feature_count))
        self.train_labels = np.array(y_labels).reshape((m, self.output_count))
        print("Converted curves to features and labels.")

        self.model.fit(self.train_data, self.train_labels, nb_epoch=50)
        print("Training complete.")

        pass

    def train(self, motion_curve, pressed_button_index):
        print('Training')

        x = self.__map_motion_curve_to_feature(motion_curve)
        y = self.__map_curve_to_final_grid_position(motion_curve)

        self.train_data = np.vstack([self.train_data, x])
        self.train_labels = np.vstack([self.train_labels, y])

        print(self.train_data)
        print(self.train_labels)

        self.model.fit(self.train_data, self.train_labels)

    def predict(self, motion_curve):
        x = self.__map_motion_curve_to_feature(motion_curve).reshape((1, self.curve_sample_count * 2))
        r = self.model.predict(x).flatten().reshape((self.x_size, self.y_size))
        self.emit(Events.PREDICTION, r)

        # print(r)
        # maxIndex = np.argmax(r)
        # maxValue = r[maxIndex]
        #
        # if maxValue < 0.5:
        #     print('Too uncertain.')
        # else:
        #     print("{0:.2f} sure of {1:1d}".format(maxValue, maxIndex))

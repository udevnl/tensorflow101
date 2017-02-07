import numpy as np
from PyQt4 import QtCore

from predict_mouse.components.shared import Events


class Collector(QtCore.QObject):

    def __init__(self, count):
        super().__init__()
        self.buffer = np.zeros((count, 2), dtype=np.float32)
        self.count = count
        self.index = 0

    def post_mouse_data(self, x, y):
        if self.index == 0:
            self.buffer[self.index][0] = x
            self.buffer[self.index][1] = y
        elif self.index < self.count:
            self.buffer[self.index][0] = x
            self.buffer[self.index][1] = y

        if self.index < self.count:
            self.index += 1

            if self.index == self.count:
                self.emit(Events.MOTION_STARTED, self.buffer)

    def post_button_click(self, button_id):
        if self.index == self.count:
            self.emit(Events.MOTION_COMPLETED, self.buffer, button_id)
        else:
            print('Not enough data to forward')
        self.index = 0
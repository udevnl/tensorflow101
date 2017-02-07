import numpy as np
from PyQt4 import QtCore

from predict_mouse.components.shared import Events


class FullCollector(QtCore.QObject):

    def __init__(self):
        super().__init__()
        self.buffer = []
        self.index = 0

    def post_mouse_data(self, x, y):
        self.buffer.append([x, y])

    def post_button_click(self, button_id):
        self.emit(Events.MOTION_COMPLETED, np.array(self.buffer))
        self.buffer = []
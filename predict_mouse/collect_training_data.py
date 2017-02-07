import sys

from PyQt4 import QtGui, QtCore

from predict_mouse.components.collector import Collector
from predict_mouse.components.full_collector import FullCollector
from predict_mouse.components.shared import Events
from predict_mouse.components.window import MainWindow

BUTTONS_X = 8
BUTTONS_Y = 8
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


def handle_sample(buffer):
    with open('mouse_train_data.csv', 'a') as f:
        csv_string = ','.join(['%.0f' % num for num in buffer.flatten()])
        f.write(csv_string)
        f.write('\n')

def main():
    app = QtGui.QApplication(sys.argv)

    main_window = MainWindow(BUTTONS_X, BUTTONS_Y)
    collector = FullCollector()

    QtCore.QObject.connect(main_window, Events.MOUSE_MOVED, collector.post_mouse_data)
    QtCore.QObject.connect(main_window, Events.MOUSE_CLICKED, collector.post_button_click)
    QtCore.QObject.connect(collector, Events.MOTION_COMPLETED, handle_sample)

    app.installEventFilter(main_window)

    main_window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

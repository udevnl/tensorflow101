import sys
from PyQt4 import QtGui, QtCore

from predict_mouse.collector import Collector
from predict_mouse.predictor import Predictor
from predict_mouse.shared import Events
from predict_mouse.window import MainWindow


def main():
    app = QtGui.QApplication(sys.argv)

    main_window = MainWindow(5, 4)
    collector = Collector(20)
    predictor = Predictor(5 * 4, 20)

    QtCore.QObject.connect(main_window, Events.MOUSE_MOVED, collector.post_mouse_data)
    QtCore.QObject.connect(main_window, Events.MOUSE_CLICKED, collector.post_button_click)
    QtCore.QObject.connect(collector, Events.MOTION_STARTED, predictor.predict)
    QtCore.QObject.connect(collector, Events.MOTION_COMPLETED, predictor.train)

    app.installEventFilter(main_window)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
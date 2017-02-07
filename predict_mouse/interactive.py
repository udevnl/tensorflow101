import sys

from PyQt4 import QtGui, QtCore

from predict_mouse.components.collector import Collector
from predict_mouse.components.logisticgridpredictor import LogisticGridPredictor
from predict_mouse.components.plot_window import PlotWindow
from predict_mouse.components.shared import Events
from predict_mouse.components.window import MainWindow

BUTTONS_X = 5
BUTTONS_Y = 4
GRID_X = 7
GRID_Y = 7
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PREDICTION_SAMPLES = 50
SIGMA = 0.5


def main():
    app = QtGui.QApplication(sys.argv)

    main_window = MainWindow(BUTTONS_X, BUTTONS_Y)
    plot_window = PlotWindow()
    collector = Collector(PREDICTION_SAMPLES)
    # predictor = LogisticButtonPredictor(BUTTONS_X * BUTTONS_Y, PREDICTION_SAMPLES)
    predictor = LogisticGridPredictor(GRID_X, GRID_Y, WINDOW_WIDTH, WINDOW_HEIGHT, SIGMA, PREDICTION_SAMPLES)

    QtCore.QObject.connect(main_window, Events.MOUSE_MOVED, collector.post_mouse_data)
    QtCore.QObject.connect(main_window, Events.MOUSE_CLICKED, collector.post_button_click)
    QtCore.QObject.connect(collector, Events.MOTION_STARTED, predictor.predict)
    QtCore.QObject.connect(collector, Events.MOTION_COMPLETED, predictor.train)
    QtCore.QObject.connect(predictor, Events.PREDICTION, plot_window.plot)

    app.installEventFilter(main_window)

    main_window.show()
    plot_window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
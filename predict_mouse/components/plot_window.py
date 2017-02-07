import matplotlib.pyplot as plt
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class PlotWindow(QtGui.QWidget):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # Just some button connected to `plot` method
        # self.button = QtGui.QPushButton('Plot')
        # self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)

    def plot(self, data_2d):
        ''' plot some random stuff '''

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        print(data_2d)

        # plot data
        ax.imshow(data_2d, cmap='gray')
        # ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

# if __name__ == '__main__':
#     app = QtGui.QApplication(sys.argv)
#
#     main = PlotWindow()
#     main.show()
#
#     sys.exit(app.exec_())

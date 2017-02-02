from PyQt4 import QtGui, QtCore
from predict_mouse.shared import Events


class MainWindow(QtGui.QWidget):
    def __init__(self, button_count_x, button_count_y):
        super(MainWindow, self).__init__()

        self.initUI(button_count_x, button_count_y)

    def initUI(self, button_count_x, button_count_y):
        self.resize(800, 600)
        self.center()
        self.addButtons(button_count_x, button_count_y)

        self.setWindowTitle('Center')
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def addButtons(self, button_count_x, button_count_y):

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        button_number = 1

        for x in range(button_count_x):
            for y in range(button_count_y):
                button = QtGui.QPushButton(str(button_number), self)
                button.myId = button_number - 1
                button.clicked.connect(self.buttonClicked)
                grid.addWidget(button, *(x, y))
                button_number += 1

    def buttonClicked(self):
        sender = self.sender()
        self.emit(Events.MOUSE_CLICKED, sender.myId)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() == QtCore.Qt.NoButton:
                pos = event.pos()
                self.emit(Events.MOUSE_MOVED, pos.x(), pos.y())
            else:
                pass # do other stuff

        return QtGui.QMainWindow.eventFilter(self, source, event)

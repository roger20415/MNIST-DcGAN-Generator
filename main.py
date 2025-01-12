from PyQt5.QtWidgets import QApplication
from gui import Gui
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Gui()
    window.show()
    sys.exit(app.exec_())
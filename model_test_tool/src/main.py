import sys
import os
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

# Enable high DPI scaling before creating QApplication
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
sys.argv += ['--disable-web-security']

# Import after setting environment variable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

def main():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application font to scale with DPI
    font = app.font()
    font.setPointSize(font.pointSize() * 1)  # Maintain standard size
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
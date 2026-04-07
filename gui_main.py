import sys
from PySide6.QtWidgets import QApplication
from evo.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Optional: Apply some global styling
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

"""
Project Root: main.py
Starts the Smart Bookstore app.
Run from the project root with: python main.py
"""
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
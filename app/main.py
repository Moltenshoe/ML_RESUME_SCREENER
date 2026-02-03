import sys
import os
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow


class ResumeApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

        self.window = MainWindow()
        self.apply_theme("dark")
        self.window.show()

    def apply_theme(self, theme_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        theme_path = os.path.join(base_dir, "themes", f"{theme_name}.qss")

        with open(theme_path, "r") as f:
            self.setStyleSheet(f.read())


if __name__ == "__main__":
    app = ResumeApp(sys.argv)
    sys.exit(app.exec())

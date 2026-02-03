import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel,
    QFileDialog, QMessageBox, QApplication
)
from PySide6.QtCore import Qt

from app.services.model_service import ModelService


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Resume Intelligence")
        self.setMinimumSize(800, 600)

        self.model_service = ModelService()
        self.current_theme = "dark"

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("Resume Intelligence")
        title.setObjectName("title")

        self.theme_button = QPushButton("☀ Light Mode")
        self.theme_button.setFixedWidth(140)
        self.theme_button.clicked.connect(self.toggle_theme)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.theme_button)

        # Text Area
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Paste resume text here...")

        # Buttons
        button_layout = QHBoxLayout()

        load_btn = QPushButton("Load Resume (.txt)")
        predict_btn = QPushButton("Analyze Resume")

        load_btn.clicked.connect(self.load_file)
        predict_btn.clicked.connect(self.predict_resume)

        button_layout.addWidget(load_btn)
        button_layout.addWidget(predict_btn)

        # Result Label
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setObjectName("result")

        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.text_area)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Resume", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                self.text_area.setText(f.read())

    def predict_resume(self):
        text = self.text_area.toPlainText().strip()

        if not text:
            QMessageBox.warning(self, "Warning", "Please enter resume text.")
            return

        results = self.model_service.predict_top3(text)

        display_text = "Top Matches:\n\n"

        for i, (label, score) in enumerate(results, start=1):
            display_text += f"{i}. {label}  (score: {score})\n"

        self.result_label.setText(display_text)


        from PySide6.QtWidgets import QApplication

    def toggle_theme(self):
        app = QApplication.instance()

        if self.current_theme == "dark":
            app.apply_theme("light")
            self.current_theme = "light"
            self.theme_button.setText("🌙 Dark Mode")
        else:
             app.apply_theme("dark")
             self.current_theme = "dark"
             self.theme_button.setText("☀ Light Mode")

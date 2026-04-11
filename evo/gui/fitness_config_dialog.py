import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QPlainTextEdit, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtGui import QFont

DEFAULT_FITNESS_CODE = """def custom_fitness(y_true, y_pred, n_total, n_selected, metrics):
    \"\"\"
    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        n_total: Total number of features in the dataset
        n_selected: Number of features selected by this individual
        metrics: Dict containing 'mcc', 'acc', 'f1', 'prec', 'recall'
    
    Returns:
        float: The fitness score (higher is better).
    \"\"\"
    # Example: MCC with a 1% penalty per selected feature
    penalty = 0.01 * (n_selected / n_total)
    return metrics['mcc'] - penalty
"""

class FitnessConfigDialog(QDialog):
    def __init__(self, current_code=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Fitness Function Configuration")
        self.resize(700, 500)
        self.code = current_code or DEFAULT_FITNESS_CODE
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Implement your custom fitness logic in Python:"))
        
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(self.code)
        # Use a monospace font for code
        font = QFont("Consolas", 10)
        if font.exactMatch(): self.editor.setFont(font)
        layout.addWidget(self.editor)
        
        btn_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load .evofitness")
        load_btn.clicked.connect(self._on_load)
        
        save_btn = QPushButton("Save .evofitness")
        save_btn.clicked.connect(self._on_save)
        
        verify_btn = QPushButton("Verify Syntax")
        verify_btn.clicked.connect(self._on_verify)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #7fbf7f; color: #11151b; font-weight: bold;")
        apply_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(verify_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def get_code(self):
        return self.editor.toPlainText()

    def _on_verify(self):
        code = self.editor.toPlainText()
        try:
            compile(code, '<string>', 'exec')
            QMessageBox.information(self, "Syntax OK", "Python code is valid.")
        except Exception as e:
            QMessageBox.critical(self, "Syntax Error", f"Error in code:\\n{e}")

    def _on_save(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Fitness Function", "", "Evo Fitness (*.evofitness)")
        if file_path:
            if not file_path.endswith(".evofitness"): file_path += ".evofitness"
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.editor.toPlainText())
                QMessageBox.information(self, "Success", f"Fitness saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Fitness Function", "", "Evo Fitness (*.evofitness)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

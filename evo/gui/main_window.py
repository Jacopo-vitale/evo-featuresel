import sys
import logging
import os
import pandas as pd
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QFileDialog, 
    QSpinBox, QDoubleSpinBox, QPlainTextEdit, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Slot

from evo.gui.widgets import MplCanvas
from evo.gui.worker import EvolutionWorker
from evo.gui.logger import QtLoggingHandler
from evo.utils import Setup

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 evo-featuresel Dashboard")
        self.resize(1100, 800)

        # Evolution stats for plotting
        self.history_gen = []
        self.history_best = []
        self.history_avg = []

        self._init_ui()
        self._setup_logging()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- 📂 Dataset Selection ---
        dataset_group = QGroupBox("📁 Dataset Configuration")
        dataset_layout = QFormLayout()
        
        self.train_path = QLineEdit("data/dataset.csv")
        self.val_path = QLineEdit("")
        self.test_path = QLineEdit("")
        
        def create_file_row(label, line_edit):
            row = QHBoxLayout()
            row.addWidget(line_edit)
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda: self._on_browse_general(line_edit))
            row.addWidget(btn)
            dataset_layout.addRow(label, row)

        create_file_row("Train CSV:", self.train_path)
        create_file_row("Validation CSV:", self.val_path)
        create_file_row("Test CSV:", self.test_path)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # --- ⚙️ Parameters & 📊 Graph ---
        middle_layout = QHBoxLayout()
        
        # Parameters
        param_group = QGroupBox("⚙️ Algorithm Parameters")
        form_layout = QFormLayout()
        
        self.pop_size = QSpinBox()
        self.pop_size.setRange(10, 1000)
        self.pop_size.setValue(50)
        
        self.generations = QSpinBox()
        self.generations.setRange(1, 100)
        self.generations.setValue(10)

        self.start_btn = QPushButton("🚀 Run Evolution")
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("🛑 Stop")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self._on_stop)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn, 2)
        btn_layout.addWidget(self.stop_btn, 1)

        form_layout.addRow("Population Size:", self.pop_size)
        form_layout.addRow("Generations:", self.generations)
        form_layout.addRow("", btn_layout)
        param_group.setLayout(form_layout)
        
        middle_layout.addWidget(param_group, 1)
        
        # Graph
        self.canvas = MplCanvas(self)
        middle_layout.addWidget(self.canvas, 3) 
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(10)
        
        layout.addLayout(middle_layout)

        # --- 📟 Console ---
        console_group = QGroupBox("📟 Integrated Console")
        console_layout = QVBoxLayout()
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: 'Consolas'; font-size: 10pt;")
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)

    def _setup_logging(self):
        self.log_handler = QtLoggingHandler()
        self.log_handler.log_signal.connect(self._append_log)
        
        # Attach to root logger or evo logger
        root_logger = logging.getLogger("evo")
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    @Slot(str)
    def _append_log(self, text):
        self.console.appendPlainText(text)
        # Auto-scroll to bottom
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _on_browse_general(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv)")
        if file_path:
            line_edit.setText(file_path)

    def _on_start(self):
        # 1. Gather inputs
        t_path = self.train_path.text()
        v_path = self.val_path.text()
        ts_path = self.test_path.text()

        if not os.path.exists(t_path):
            self.console.appendPlainText(f"❌ ERROR: Train file not found: {t_path}")
            return

        # 2. Reset Plot Data
        self.history_gen = []
        self.history_best = []
        self.history_avg = []
        self.canvas.plot_data([], [], []) # Reset view

        # 3. Threading
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        params = {
            'train_path': t_path,
            'val_path': v_path,
            'test_path': ts_path,
            'pop_size': self.pop_size.value(),
            'generations': self.generations.value(),
            'seed': 42
        }
        
        self.worker = EvolutionWorker(params)
        self.worker.generation_completed.connect(self._on_generation_update)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_stop(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.console.appendPlainText("⏳ Stopping evolution... please wait for the current generation to finish.")
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    @Slot(int, float, float)
    def _on_generation_update(self, gen, best, avg):
        self.history_gen.append(gen)
        self.history_best.append(best)
        self.history_avg.append(avg)
        self.canvas.plot_data(self.history_gen, self.history_best, self.history_avg)

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.console.appendPlainText("✅ SUCCESS: Evolution completed.")

    def _on_error(self, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.console.appendPlainText(f"❌ ERROR: {msg}")

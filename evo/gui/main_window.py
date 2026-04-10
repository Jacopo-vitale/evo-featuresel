import logging
import os
import json
from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QMessageBox
)

from evo.gui.logger import QtLoggingHandler
from evo.gui.widgets import MplCanvas
from evo.gui.worker import EvolutionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("evo-featuresel Dashboard")
        self.resize(1200, 850)

        icon_path = os.path.join("assets", "dna.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.history_gen = []
        self.history_best_f = []
        self.history_avg_f = []
        
        # New history trackers for metrics
        self.history_metrics = {
            "MCC": {"best": [], "avg": []},
            "Accuracy": {"best": [], "avg": []},
            "F1": {"best": [], "avg": []},
            "Precision": {"best": [], "avg": []},
            "Recall": {"best": [], "avg": []}
        }

        self._init_ui()
        self._setup_logging()
        self._apply_theme()
        
        # Set initial plot state
        self._update_plot()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(15)

        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QFormLayout()

        self.train_path = QLineEdit("data/dataset.csv")
        self.train_labels_path = QLineEdit("")
        self.val_path = QLineEdit("")
        self.val_labels_path = QLineEdit("")
        self.test_path = QLineEdit("")
        self.test_labels_path = QLineEdit("")

        def create_file_row(label, line_edit):
            row = QHBoxLayout()
            row.addWidget(line_edit)
            btn = QPushButton("...")
            btn.setFixedWidth(30)
            btn.clicked.connect(lambda: self._on_browse_general(line_edit))
            row.addWidget(btn)
            dataset_layout.addRow(label, row)

        create_file_row("Train Features:", self.train_path)
        create_file_row("Train Labels:", self.train_labels_path)
        create_file_row("Val Features:", self.val_path)
        create_file_row("Val Labels:", self.val_labels_path)
        create_file_row("Test Features:", self.test_path)
        create_file_row("Test Labels:", self.test_labels_path)

        dataset_group.setLayout(dataset_layout)
        sidebar.addWidget(dataset_group)

        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        
        checks_row = QHBoxLayout()
        self.pca_check = QCheckBox("Apply PCA (95% variance)")
        self.lda_check = QCheckBox("Apply LDA")
        self.pca_check.toggled.connect(lambda checked: self.lda_check.setChecked(False) if checked else None)
        self.lda_check.toggled.connect(lambda checked: self.pca_check.setChecked(False) if checked else None)
        checks_row.addWidget(self.pca_check)
        checks_row.addWidget(self.lda_check)
        preproc_layout.addLayout(checks_row)

        scaler_row = QHBoxLayout()
        scaler_row.addWidget(QLabel("Scaler:"))
        self.scaler_type = QComboBox()
        self.scaler_type.addItems(["Standard", "MinMax"])
        scaler_row.addWidget(self.scaler_type)
        preproc_layout.addLayout(scaler_row)

        preproc_group.setLayout(preproc_layout)
        sidebar.addWidget(preproc_group)

        param_group = QGroupBox("Algorithm Parameters")
        param_form = QFormLayout()

        self.pop_size = QSpinBox()
        self.pop_size.setRange(10, 1000)
        self.pop_size.setValue(50)

        self.generations = QSpinBox()
        self.generations.setRange(1, 1000)
        self.generations.setValue(10)

        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.01, 2.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(0.5)

        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(1, 10)
        self.cv_folds.setValue(1)

        param_form.addRow("Population Size:", self.pop_size)
        param_form.addRow("Generations:", self.generations)
        param_form.addRow("Random Seed:", self.seed)
        param_form.addRow("Mutation Alpha:", self.alpha)
        param_form.addRow("Outer CV Folds:", self.cv_folds)

        param_group.setLayout(param_form)
        sidebar.addWidget(param_group)

        # New: Experiment Configuration
        exp_group = QGroupBox("Experiment Configuration")
        exp_layout = QFormLayout()

        self.description = QLineEdit("Evolutionary Feature Selection Experiment")
        self.penalty = QDoubleSpinBox()
        self.penalty.setRange(0.0, 1.0)
        self.penalty.setSingleStep(0.01)
        self.penalty.setValue(0.01)
        self.penalty.setToolTip("Penalty for each selected feature as a ratio of total features.")

        self.metric_selector = QComboBox()
        self.metric_selector.addItems(["MCC", "Accuracy", "F1", "Precision", "Recall"])
        self.metric_selector.currentIndexChanged.connect(self._update_plot)

        exp_layout.addRow("Description:", self.description)
        exp_layout.addRow("Penalty Factor:", self.penalty)
        exp_layout.addRow("Display Metric:", self.metric_selector)
        exp_group.setLayout(exp_layout)
        sidebar.addWidget(exp_group)

        # New: Output Settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()

        self.exp_folder = QLineEdit("experiment")
        self.proj_prefix = QLineEdit("gui_run_")
        self.use_timestamp = QCheckBox("Append Timestamp")
        self.use_timestamp.setChecked(True)

        output_layout.addRow("Base Folder:", self.exp_folder)
        output_layout.addRow("Folder Prefix:", self.proj_prefix)
        output_layout.addRow(self.use_timestamp)
        output_group.setLayout(output_layout)
        sidebar.addWidget(output_group)

        sidebar.addStretch()

        # Control Buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)

        self.start_btn = QPushButton("Run Evolution")
        self.start_btn.setFixedHeight(45)
        self.start_btn.setStyleSheet("background-color: #7fbf7f; color: #11151b; font-weight: bold; font-size: 10pt;")
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #e5a6b6; color: #11151b; font-weight: bold;")
        self.stop_btn.clicked.connect(self._on_stop)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedHeight(45)
        self.reset_btn.setStyleSheet("background-color: #727172; color: #d8dee9; font-weight: bold;")
        self.reset_btn.clicked.connect(self._on_reset)
        
        controls_layout.addWidget(self.start_btn, 3)
        controls_layout.addWidget(self.stop_btn, 2)
        controls_layout.addWidget(self.reset_btn, 2)
        sidebar.addLayout(controls_layout)

        main_layout.addLayout(sidebar, 1)

        content = QVBoxLayout()
        content.setSpacing(10)

        self.canvas = MplCanvas(self, width=8, height=10)
        content.addWidget(self.canvas, 10)

        self.results_group = QGroupBox("Best Individual Results")
        self.results_group.setVisible(False)
        results_layout = QHBoxLayout()

        self.res_labels = {
            'fitness': QLabel("MCC: -"),
            'acc': QLabel("Acc: -"),
            'model': QLabel("Model: -"),
            'features': QLabel("Features: -")
        }
        for lbl in self.res_labels.values():
            lbl.setStyleSheet("font-weight: bold; color: #8bd3dd; font-size: 11pt;")
            results_layout.addWidget(lbl)

        self.results_group.setLayout(results_layout)
        content.addWidget(self.results_group)

        console_group = QGroupBox("Integrated Console")
        console_layout = QVBoxLayout()
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(180)
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)
        content.addWidget(console_group)

        status_bar_layout = QHBoxLayout()
        self.avg_feat_lbl = QLabel("Avg Features: -")
        self.gen_time_lbl = QLabel("Gen Time: -")
        self.eta_lbl = QLabel("ETA: -")

        for lbl in [self.avg_feat_lbl, self.gen_time_lbl, self.eta_lbl]:
            lbl.setStyleSheet("color: #aab4c3; font-size: 9pt; font-weight: bold;")
            status_bar_layout.addWidget(lbl)
            status_bar_layout.addStretch()

        content.addLayout(status_bar_layout)
        main_layout.addLayout(content, 4)

        self._create_menu_bar()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        load_action = QAction("Load Configuration...", self)
        load_action.triggered.connect(self._on_load_config)
        file_menu.addAction(load_action)

        save_action = QAction("Save Configuration...", self)
        save_action.triggered.connect(self._on_save_config)
        file_menu.addAction(save_action)

    def _get_params(self):
        return {
            'train_path': self.train_path.text(),
            'train_labels_path': self.train_labels_path.text(),
            'val_path': self.val_path.text(),
            'val_labels_path': self.val_labels_path.text(),
            'test_path': self.test_path.text(),
            'test_labels_path': self.test_labels_path.text(),
            'pop_size': self.pop_size.value(),
            'generations': self.generations.value(),
            'seed': self.seed.value(),
            'alpha': self.alpha.value(),
            'cv_folds': self.cv_folds.value(),
            'pca': self.pca_check.isChecked(),
            'lda': self.lda_check.isChecked(),
            'scaler_type': self.scaler_type.currentText(),
            'description': self.description.text(),
            'penalty_factor': self.penalty.value(),
            'experiment_folder': self.exp_folder.text(),
            'project_prefix': self.proj_prefix.text(),
            'use_timestamp': self.use_timestamp.isChecked()
        }

    def _set_params(self, params):
        self.train_path.setText(params.get('train_path', ''))
        self.train_labels_path.setText(params.get('train_labels_path', ''))
        self.val_path.setText(params.get('val_path', ''))
        self.val_labels_path.setText(params.get('val_labels_path', ''))
        self.test_path.setText(params.get('test_path', ''))
        self.test_labels_path.setText(params.get('test_labels_path', ''))
        
        self.pop_size.setValue(params.get('pop_size', 50))
        self.generations.setValue(params.get('generations', 10))
        self.seed.setValue(params.get('seed', 42))
        self.alpha.setValue(params.get('alpha', 0.5))
        self.cv_folds.setValue(params.get('cv_folds', 1))
        
        self.pca_check.setChecked(params.get('pca', False))
        self.lda_check.setChecked(params.get('lda', False))
        
        scaler = params.get('scaler_type', 'Standard')
        idx = self.scaler_type.findText(scaler)
        if idx >= 0:
            self.scaler_type.setCurrentIndex(idx)
            
        self.description.setText(params.get('description', 'Evolutionary Feature Selection Experiment'))
        self.penalty.setValue(params.get('penalty_factor', 0.01))
        self.exp_folder.setText(params.get('experiment_folder', 'experiment'))
        self.proj_prefix.setText(params.get('project_prefix', 'gui_run_'))
        self.use_timestamp.setChecked(params.get('use_timestamp', True))

    def _on_save_config(self):
        params = self._get_params()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "Evo Config Files (*.evoconf)")
        if file_path:
            if not file_path.endswith(".evoconf"):
                file_path += ".evoconf"
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=4)
                self.console.appendPlainText(f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\\n{e}")

    def _on_load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "Evo Config Files (*.evoconf);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self._set_params(params)
                self.console.appendPlainText(f"Configuration loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load configuration:\\n{e}")

    def _setup_logging(self):
        self.log_handler = QtLoggingHandler()
        self.log_handler.log_signal.connect(self._append_log)

        root_logger = logging.getLogger("evo")
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #11151b;
                color: #d8dee9;
            }
            QGroupBox {
                background-color: #171b22;
                border: 1px solid #2d3743;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
                color: #d8dee9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #f2cdcd;
            }
            QLabel {
                color: #d8dee9;
            }
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1f2630;
                border: 1px solid #3b4757;
                border-radius: 7px;
                padding: 6px 8px;
                color: #e5e9f0;
                selection-background-color: #8bd3dd;
                selection-color: #11151b;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #8bd3dd;
            }
            QPushButton {
                background-color: #2a3441;
                border: 1px solid #465466;
                border-radius: 8px;
                color: #e5e9f0;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #344153;
            }
            QPushButton:disabled {
                background-color: #20262f;
                color: #7f8a98;
                border-color: #2d3541;
            }
            QCheckBox {
                color: #cdd6e3;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #536276;
                background: #1f2630;
            }
            QCheckBox::indicator:checked {
                background: #a6d189;
                border: 1px solid #a6d189;
            }
        """)

    @Slot(str)
    def _append_log(self, text):
        self.console.appendPlainText(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _on_browse_general(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv)")
        if file_path:
            line_edit.setText(file_path)

    def _on_start(self):
        params = self._get_params()

        if not os.path.exists(params['train_path']):
            self.console.appendPlainText(f"ERROR: Train file not found: {params['train_path']}")
            return

        self._set_inputs_enabled(False)
        self.stop_btn.setEnabled(True)

        self.history_gen = []
        self.history_best_f = []
        self.history_avg_f = []
        for m in self.history_metrics:
            self.history_metrics[m]["best"] = []
            self.history_metrics[m]["avg"] = []
            
        self.canvas.plot_multi_data([], ([], []), ([], []))
        self.canvas.plot_stats({})
        self.results_group.setVisible(False)

        self.worker = EvolutionWorker(params)
        self.worker.generation_completed.connect(self._on_generation_update)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_stop(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.console.appendPlainText("Stopping evolution. Waiting for the current generation to finish.")
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def _set_inputs_enabled(self, enabled: bool):
        # Dataset
        self.train_path.setEnabled(enabled)
        self.train_labels_path.setEnabled(enabled)
        self.val_path.setEnabled(enabled)
        self.val_labels_path.setEnabled(enabled)
        self.test_path.setEnabled(enabled)
        self.test_labels_path.setEnabled(enabled)
        
        # Preprocessing
        self.pca_check.setEnabled(enabled)
        self.lda_check.setEnabled(enabled)
        self.scaler_type.setEnabled(enabled)
        
        # Algorithm
        self.pop_size.setEnabled(enabled)
        self.generations.setEnabled(enabled)
        self.seed.setEnabled(enabled)
        self.alpha.setEnabled(enabled)
        self.cv_folds.setEnabled(enabled)
        
        # Experiment
        self.description.setEnabled(enabled)
        self.penalty.setEnabled(enabled)
        self.metric_selector.setEnabled(enabled)
        
        # Output
        self.exp_folder.setEnabled(enabled)
        self.proj_prefix.setEnabled(enabled)
        self.use_timestamp.setEnabled(enabled)
        
        # Main Buttons
        self.start_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)

    def _on_reset(self):
        # 1. Reset Dataset paths
        self.train_path.setText("data/dataset.csv")
        self.train_labels_path.setText("")
        self.val_path.setText("")
        self.val_labels_path.setText("")
        self.test_path.setText("")
        self.test_labels_path.setText("")

        # 2. Reset Preprocessing
        self.pca_check.setChecked(False)
        self.lda_check.setChecked(False)
        self.scaler_type.setCurrentIndex(0) # Standard

        # 3. Reset Algorithm Params
        self.pop_size.setValue(50)
        self.generations.setValue(10)
        self.seed.setValue(42)
        self.alpha.setValue(0.5)
        self.cv_folds.setValue(1)

        # 4. Reset Experiment & Output
        self.description.setText("Evolutionary Feature Selection Experiment")
        self.penalty.setValue(0.01)
        self.metric_selector.setCurrentIndex(0) # MCC
        self.exp_folder.setText("experiment")
        self.proj_prefix.setText("gui_run_")
        self.use_timestamp.setChecked(True)

        # 5. Clear Histories and Plots
        self.history_gen = []
        self.history_best_f = []
        self.history_avg_f = []
        for m in self.history_metrics:
            self.history_metrics[m]["best"] = []
            self.history_metrics[m]["avg"] = []
            
        self.canvas.plot_multi_data([], ([], []), ([], []))
        self.canvas.plot_stats({})
        self._update_plot()
        self.results_group.setVisible(False)
        self.avg_feat_lbl.setText("Avg Features: -")
        self.gen_time_lbl.setText("Gen Time: -")
        self.eta_lbl.setText("ETA: -")
        for lbl in self.res_labels.values():
            lbl.setText(lbl.text().split(":")[0] + ": -")
        self.console.clear()
        self.console.appendPlainText("UI Reset to defaults.")
        
        # 6. Re-enable inputs and reset buttons
        self._set_inputs_enabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @Slot(int, float, float, dict)
    def _on_generation_update(self, gen, best_f, avg_f, stats):
        self.history_gen.append(gen)
        self.history_best_f.append(best_f)
        self.history_avg_f.append(avg_f)
        
        # Store other metrics from stats
        self.history_metrics["MCC"]["avg"].append(stats.get('avg_mcc', 0))
        self.history_metrics["Accuracy"]["avg"].append(stats.get('avg_acc', 0))
        self.history_metrics["F1"]["avg"].append(stats.get('avg_f1', 0))
        self.history_metrics["Precision"]["avg"].append(stats.get('avg_prec', 0))
        self.history_metrics["Recall"]["avg"].append(stats.get('avg_recall', 0))
        
        self.history_metrics["MCC"]["best"].append(stats.get('best_mcc', 0))
        self.history_metrics["Accuracy"]["best"].append(stats.get('best_acc', 0))
        self.history_metrics["F1"]["best"].append(stats.get('best_f1', 0))
        self.history_metrics["Precision"]["best"].append(stats.get('best_prec', 0))
        self.history_metrics["Recall"]["best"].append(stats.get('best_recall', 0))

        self._update_plot()
        self.canvas.plot_stats(stats)

        self.avg_feat_lbl.setText(f"Avg Features: {stats.get('avg_features', 0):.1f}")
        if 'gen_time' in stats:
            self.gen_time_lbl.setText(f"Gen Time: {stats['gen_time']:.2f}s")
        if 'eta' in stats:
            eta = stats['eta']
            self.eta_lbl.setText(f"ETA: {int(eta // 60)}m {int(eta % 60)}s")

    def _update_plot(self):
        metric_name = self.metric_selector.currentText()
        fitness_data = (self.history_best_f, self.history_avg_f)
        metric_data = (self.history_metrics[metric_name]["best"], self.history_metrics[metric_name]["avg"])
        
        self.canvas.plot_multi_data(self.history_gen, fitness_data, metric_data, metric_name)

    @Slot(dict)
    def _on_finished(self, results):
        self._set_inputs_enabled(True)
        self.stop_btn.setEnabled(False)
        self.console.appendPlainText("SUCCESS: Evolution completed.")
        output_folder = results.get('output_folder')
        if output_folder:
            self.console.appendPlainText(f"Output folder: {output_folder}")

        if 'std_fitness' in results:
            self.res_labels['fitness'].setText(f"MCC: {results['fitness']:.4f} ± {results['std_fitness']:.4f}")
        else:
            self.res_labels['fitness'].setText(f"MCC: {results['fitness']:.4f}")

        self.res_labels['acc'].setText(f"Acc: {results['acc']:.4f}")
        self.res_labels['model'].setText(f"Model: {results['model_type']}")
        self.res_labels['features'].setText(f"Features: {results['features_count']:.1f}")
        self.results_group.setVisible(True)

    def _on_error(self, msg):
        self._set_inputs_enabled(True)
        self.stop_btn.setEnabled(False)
        self.console.appendPlainText(f"ERROR: {msg}")

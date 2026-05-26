import logging
import os
import json
from PySide6.QtCore import Slot, Qt
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
    QMessageBox,
    QScrollArea
)

from evo.gui.logger import QtLoggingHandler
from evo.gui.widgets import MplCanvas
from evo.gui.worker import EvolutionWorker
from evo.gui.individual_config_dialog import IndividualConfigDialog
from evo.gui.fitness_config_dialog import FitnessConfigDialog, DEFAULT_FITNESS_CODE
from evo.utils import get_default_individual_config


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
        
        self.individual_config = get_default_individual_config()
        self.fitness_code = DEFAULT_FITNESS_CODE
        self.ui_scale = 1.0 # Default scale
        
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
        
        # Track saved state
        self._last_saved_params = self._get_params()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Sidebar with ScrollArea ---
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QScrollArea.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFixedWidth(340)
        
        sidebar_content = QWidget()
        sidebar = QVBoxLayout(sidebar_content)
        sidebar.setSpacing(12)
        sidebar.setContentsMargins(0, 0, 10, 0)

        # 1. Dataset Group
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QFormLayout()
        dataset_layout.setSpacing(5)

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

        # 2. Preprocessing Group
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        preproc_layout.setSpacing(8)
        
        checks_row = QHBoxLayout()
        self.pca_check = QCheckBox("Apply PCA (95%)")
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

        # 3. Algorithm Group
        param_group = QGroupBox("Algorithm Parameters")
        param_form = QFormLayout()
        param_form.setSpacing(5)

        self.pop_size = QSpinBox(); self.pop_size.setRange(10, 1000); self.pop_size.setValue(50)
        self.generations = QSpinBox(); self.generations.setRange(1, 1000); self.generations.setValue(10)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999); self.seed.setValue(42)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.01, 2.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.5)
        self.cv_folds = QSpinBox(); self.cv_folds.setRange(1, 10); self.cv_folds.setValue(1)

        param_form.addRow("Pop Size:", self.pop_size)
        param_form.addRow("Generations:", self.generations)
        param_form.addRow("Seed:", self.seed)
        param_form.addRow("Mut Alpha:", self.alpha)
        param_form.addRow("CV Folds:", self.cv_folds)

        param_group.setLayout(param_form)
        sidebar.addWidget(param_group)

        # 4. Experiment Group
        exp_group = QGroupBox("Experiment Settings")
        exp_layout = QFormLayout()
        exp_layout.setSpacing(5)

        self.description = QLineEdit("Evo Experiment")
        self.penalty = QDoubleSpinBox(); self.penalty.setRange(0.0, 1.0); self.penalty.setSingleStep(0.01); self.penalty.setValue(0.01)
        self.metric_selector = QComboBox(); self.metric_selector.addItems(["MCC", "Accuracy", "F1", "Precision", "Recall"])
        self.metric_selector.currentIndexChanged.connect(self._update_plot)

        exp_layout.addRow("Desc:", self.description)
        exp_layout.addRow("Penalty:", self.penalty)
        exp_layout.addRow("Display:", self.metric_selector)
        exp_group.setLayout(exp_layout)
        sidebar.addWidget(exp_group)

        # 5. Output Group
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        output_layout.setSpacing(5)

        self.exp_folder = QLineEdit("experiment")
        self.proj_prefix = QLineEdit("gui_run_")
        self.use_timestamp = QCheckBox("Timestamp folder")
        self.use_timestamp.setChecked(True)

        output_layout.addRow("Root:", self.exp_folder)
        output_layout.addRow("Prefix:", self.proj_prefix)
        output_layout.addRow(self.use_timestamp)
        output_group.setLayout(output_layout)
        sidebar.addWidget(output_group)

        sidebar.addStretch()

        # Control Buttons (Fixed at bottom of sidebar)
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 5, 0, 0)
        controls_layout.setSpacing(5)

        self.start_btn = QPushButton("Run")
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("background-color: #7fbf7f; color: #11151b; font-weight: bold;")
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(40); self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #e5a6b6; color: #11151b; font-weight: bold;")
        self.stop_btn.clicked.connect(self._on_stop)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedHeight(40)
        self.reset_btn.clicked.connect(self._on_reset)
        
        controls_layout.addWidget(self.start_btn, 2)
        controls_layout.addWidget(self.stop_btn, 1)
        controls_layout.addWidget(self.reset_btn, 1)
        sidebar.addWidget(controls_container)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll, 0) # 0 means don't stretch sidebar

        # --- Content Area ---
        content = QVBoxLayout()
        content.setSpacing(10)

        self.canvas = MplCanvas(self, width=8, height=10)
        content.addWidget(self.canvas, 10)

        self.results_group = QGroupBox("Best Results")
        self.results_group.setVisible(False)
        results_layout = QHBoxLayout()
        self.res_labels = {
            'fitness': QLabel("MCC: -"), 'acc': QLabel("Acc: -"),
            'model': QLabel("Model: -"), 'features': QLabel("Features: -")
        }
        for lbl in self.res_labels.values():
            lbl.setStyleSheet("font-weight: bold; color: #8bd3dd; font-size: 10pt;")
            results_layout.addWidget(lbl)
        self.results_group.setLayout(results_layout)
        content.addWidget(self.results_group)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        content.addWidget(self.console)

        status_bar_layout = QHBoxLayout()
        self.avg_feat_lbl = QLabel("Avg Features: -")
        self.gen_time_lbl = QLabel("Gen Time: -")
        self.eta_lbl = QLabel("ETA: -")
        for lbl in [self.avg_feat_lbl, self.gen_time_lbl, self.eta_lbl]:
            lbl.setStyleSheet("color: #aab4c3; font-size: 9pt;")
            status_bar_layout.addWidget(lbl)
            status_bar_layout.addStretch()
        content.addLayout(status_bar_layout)

        main_layout.addLayout(content, 1) # Content stretches
        self._create_menu_bar()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("File")
        load_action = QAction("Load Config...", self); load_action.triggered.connect(self._on_load_config); file_menu.addAction(load_action)
        save_action = QAction("Save Config...", self); save_action.triggered.connect(self._on_save_config); file_menu.addAction(save_action)
        file_menu.addSeparator()
        reset_action = QAction("Reset UI", self); reset_action.triggered.connect(self._on_reset); file_menu.addAction(reset_action)
        exit_action = QAction("Exit", self); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        
        # Edit Menu
        edit_menu = menu_bar.addMenu("Edit")
        config_menu = edit_menu.addMenu("Configuration")
        
        ind_config_action = QAction("Individual...", self); ind_config_action.triggered.connect(self._on_config_individual); config_menu.addAction(ind_config_action)
        fit_config_action = QAction("Fitness...", self); fit_config_action.triggered.connect(self._on_config_fitness); config_menu.addAction(fit_config_action)

        # View Menu (UI Scaling)
        view_menu = menu_bar.addMenu("View")
        scale_menu = view_menu.addMenu("UI Scale")
        s_action = QAction("Small (80%)", self); s_action.triggered.connect(lambda: self._on_change_ui_scale(0.8)); scale_menu.addAction(s_action)
        n_action = QAction("Normal (100%)", self); n_action.triggered.connect(lambda: self._on_change_ui_scale(1.0)); scale_menu.addAction(n_action)
        l_action = QAction("Large (120%)", self); l_action.triggered.connect(lambda: self._on_change_ui_scale(1.2)); scale_menu.addAction(l_action)

        # Help Menu
        help_menu = menu_bar.addMenu("?")
        about_action = QAction("About...", self); about_action.triggered.connect(self._on_about); help_menu.addAction(about_action)

    def _on_change_ui_scale(self, scale):
        self.ui_scale = scale
        self._apply_theme()
        self.console.appendPlainText(f"UI Scale changed to {int(scale*100)}%")

    def _on_config_individual(self):
        dialog = IndividualConfigDialog(self.individual_config, self)
        if dialog.exec():
            self.individual_config = dialog.get_updated_config()
            self.console.appendPlainText("Individual configuration updated.")

    def _on_config_fitness(self):
        dialog = FitnessConfigDialog(self.fitness_code, self)
        if dialog.exec():
            self.fitness_code = dialog.get_code()
            self.console.appendPlainText("Custom fitness function updated.")

    def _on_about(self):
        QMessageBox.about(self, "About evo-featuresel", "<h3>evo-featuresel Dashboard</h3><p>Optimized Evolutionary Feature Selection.</p>")

    def closeEvent(self, event):
        # 1. Stop worker if running
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait() # Ensure it finishes before app exits

        # 2. Check for unsaved changes
        if self._get_params() == getattr(self, '_last_saved_params', None):
            event.accept()
            return
        reply = QMessageBox.question(self, "Save Config?", "Save changes before exiting?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Save:
            self._on_save_config()
            if self._get_params() == getattr(self, '_last_saved_params', None): event.accept()
            else: event.ignore()
        elif reply == QMessageBox.Cancel: event.ignore()
        else: event.accept()

    def _get_params(self):
        return {
            'train_path': self.train_path.text(), 'train_labels_path': self.train_labels_path.text(),
            'val_path': self.val_path.text(), 'val_labels_path': self.val_labels_path.text(),
            'test_path': self.test_path.text(), 'test_labels_path': self.test_labels_path.text(),
            'pop_size': self.pop_size.value(), 'generations': self.generations.value(),
            'seed': self.seed.value(), 'alpha': self.alpha.value(), 'cv_folds': self.cv_folds.value(),
            'pca': self.pca_check.isChecked(), 'lda': self.lda_check.isChecked(),
            'scaler_type': self.scaler_type.currentText(), 'description': self.description.text(),
            'penalty_factor': self.penalty.value(), 'experiment_folder': self.exp_folder.text(),
            'project_prefix': self.proj_prefix.text(), 'use_timestamp': self.use_timestamp.isChecked(),
            'individual_config': self.individual_config,
            'fitness_code': self.fitness_code
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
        if idx >= 0: self.scaler_type.setCurrentIndex(idx)
        self.description.setText(params.get('description', 'Evo Experiment'))
        self.penalty.setValue(params.get('penalty_factor', 0.01))
        self.exp_folder.setText(params.get('experiment_folder', 'experiment'))
        self.proj_prefix.setText(params.get('project_prefix', 'gui_run_'))
        self.use_timestamp.setChecked(params.get('use_timestamp', True))
        if 'individual_config' in params: self.individual_config = params['individual_config']
        if 'fitness_code' in params: self.fitness_code = params['fitness_code']

    def _on_save_config(self):
        params = self._get_params()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "Evo Config (*.evoconf)")
        if file_path:
            if not file_path.endswith(".evoconf"): file_path += ".evoconf"
            try:
                with open(file_path, 'w', encoding='utf-8') as f: json.dump(params, f, indent=4)
                self.console.appendPlainText(f"Saved: {file_path}")
                self._last_saved_params = params
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to save:\\n{e}")

    def _on_load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "Evo Config (*.evoconf);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: params = json.load(f)
                self._set_params(params)
                self.console.appendPlainText(f"Loaded: {file_path}")
                self._last_saved_params = self._get_params()
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load:\\n{e}")

    def _setup_logging(self):
        self.log_handler = QtLoggingHandler()
        self.log_handler.log_signal.connect(self._append_log)
        root_logger = logging.getLogger("evo")
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

    def _apply_theme(self):
        base_font = int(10 * self.ui_scale)
        group_font = int(11 * self.ui_scale)
        title_font = int(9 * self.ui_scale)
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: #11151b; color: #d8dee9; font-size: {base_font}pt; }}
            QGroupBox {{ background-color: #171b22; border: 1px solid #2d3743; border-radius: 8px; margin-top: 12px; padding-top: 10px; font-weight: 600; color: #d8dee9; font-size: {group_font}pt; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #f2cdcd; }}
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background-color: #1f2630; border: 1px solid #3b4757; border-radius: 5px; padding: 4px 6px; color: #e5e9f0; }}
            QPushButton {{ background-color: #2a3441; border: 1px solid #465466; border-radius: 6px; color: #e5e9f0; padding: 6px 10px; }}
            QPushButton:hover {{ background-color: #344153; }}
            QCheckBox {{ spacing: 5px; }}
            QScrollBar:vertical {{ border: none; background: #11151b; width: 10px; }}
            QScrollBar::handle:vertical {{ background: #3b4757; min-height: 20px; border-radius: 5px; }}
        """)

    @Slot(str)
    def _append_log(self, text):
        self.console.appendPlainText(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _on_browse_general(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv)")
        if file_path: line_edit.setText(file_path)

    def _on_start(self):
        params = self._get_params()
        if not os.path.exists(params['train_path']):
            self.console.appendPlainText(f"ERROR: Train file not found: {params['train_path']}")
            return
        self._set_inputs_enabled(False)
        self.stop_btn.setEnabled(True)
        self.history_gen = []; self.history_best_f = []; self.history_avg_f = []
        for m in self.history_metrics: self.history_metrics[m]["best"] = []; self.history_metrics[m]["avg"] = []
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
            self.console.appendPlainText("Stopping evolution...")
            self.worker.stop(); self.stop_btn.setEnabled(False)

    def _set_inputs_enabled(self, enabled: bool):
        for w in [self.train_path, self.train_labels_path, self.val_path, self.val_labels_path, self.test_path, self.test_labels_path,
                  self.pca_check, self.lda_check, self.scaler_type, self.pop_size, self.generations, self.seed, self.alpha, self.cv_folds,
                  self.description, self.penalty, self.metric_selector, self.exp_folder, self.proj_prefix, self.use_timestamp,
                  self.start_btn, self.reset_btn]:
            w.setEnabled(enabled)

    def _on_reset(self):
        self._set_params({'train_path': 'data/dataset.csv', 'pop_size': 50, 'generations': 10, 'seed': 42, 'alpha': 0.5, 'cv_folds': 1, 'penalty_factor': 0.01})
        self.history_gen = []; self.history_best_f = []; self.history_avg_f = []
        for m in self.history_metrics: self.history_metrics[m]["best"] = []; self.history_metrics[m]["avg"] = []
        self.canvas.plot_multi_data([], ([], []), ([], []))
        self.canvas.plot_stats({})
        self.results_group.setVisible(False)
        self.console.clear(); self.console.appendPlainText("UI Reset.")
        self._set_inputs_enabled(True); self.stop_btn.setEnabled(False)

    @Slot(int, float, float, dict)
    def _on_generation_update(self, gen, best_f, avg_f, stats):
        self.history_gen.append(gen); self.history_best_f.append(best_f); self.history_avg_f.append(avg_f)
        for m in ["MCC", "Accuracy", "F1", "Precision", "Recall"]:
            self.history_metrics[m]["avg"].append(stats.get(f'avg_{m.lower()}', 0))
            self.history_metrics[m]["best"].append(stats.get(f'best_{m.lower()}', 0))
        self._update_plot(); self.canvas.plot_stats(stats)
        self.avg_feat_lbl.setText(f"Avg Feat: {stats.get('avg_features', 0):.1f}")
        if 'gen_time' in stats: self.gen_time_lbl.setText(f"Gen: {stats['gen_time']:.2f}s")
        if 'eta' in stats: self.eta_lbl.setText(f"ETA: {int(stats['eta']//60)}m {int(stats['eta']%60)}s")

    def _update_plot(self):
        m = self.metric_selector.currentText()
        self.canvas.plot_multi_data(self.history_gen, (self.history_best_f, self.history_avg_f), (self.history_metrics[m]["best"], self.history_metrics[m]["avg"]), m)

    @Slot(dict)
    def _on_finished(self, results):
        self._set_inputs_enabled(True); self.stop_btn.setEnabled(False)
        self.console.appendPlainText(f"FINISHED. Results in: {results.get('output_folder')}")
        self.res_labels['fitness'].setText(f"MCC: {results['fitness']:.4f}")
        self.res_labels['acc'].setText(f"Acc: {results['acc']:.4f}")
        self.res_labels['model'].setText(f"Model: {results['model_type']}")
        self.res_labels['features'].setText(f"Feat: {results['features_count']:.1f}")
        self.results_group.setVisible(True)

    def _on_error(self, msg):
        self._set_inputs_enabled(True); self.stop_btn.setEnabled(False)
        self.console.appendPlainText(f"ERROR: {msg}")

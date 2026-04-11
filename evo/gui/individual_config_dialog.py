import json
import os
import copy
import importlib
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, 
    QSpinBox, QLabel, QFormLayout, QScrollArea, QWidget, 
    QPushButton, QFileDialog, QMessageBox, QComboBox, 
    QListWidget, QListWidgetItem, QStackedWidget, QSplitter,
    QLineEdit
)
from PySide6.QtCore import Qt
from evo.models_registry import reload_registry, USER_MODELS_PATH

class NewModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Model")
        self.resize(500, 400)
        self.detected_params = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.name_edit = QLineEdit(); self.name_edit.setPlaceholderText("e.g. GaussianNB")
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText("e.g. sklearn.naive_bayes.GaussianNB")
        
        verify_btn = QPushButton("Verify & Discover Parameters")
        verify_btn.clicked.connect(self._on_verify)
        
        form.addRow("Model Name:", self.name_edit)
        form.addRow("Import Path:", self.path_edit)
        form.addRow(verify_btn)
        
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_content = QWidget()
        self.param_layout = QVBoxLayout(self.param_content)
        self.param_scroll.setWidget(self.param_content)
        self.param_scroll.setVisible(False)
        
        layout.addLayout(form)
        layout.addWidget(QLabel("Select parameters to optimize:"))
        layout.addWidget(self.param_scroll)
        
        btns = QHBoxLayout()
        self.add_btn = QPushButton("Add to Registry"); self.add_btn.setEnabled(False); self.add_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel"); cancel_btn.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(self.add_btn); btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _on_verify(self):
        path = self.path_edit.text().strip()
        if not path: return
        try:
            module_name, class_name = path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            instance = model_class()
            self.detected_params = instance.get_params()
            self._populate_params()
            self.add_btn.setEnabled(True)
            QMessageBox.information(self, "Success", f"Found {len(self.detected_params)} parameters.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not load model:\\n{e}")

    def _populate_params(self):
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        self.param_checks = {}
        for p_name, p_val in sorted(self.detected_params.items()):
            row = QHBoxLayout()
            cb = QCheckBox(p_name)
            row.addWidget(cb)
            
            p_type = QComboBox()
            p_type.addItems(["int", "float", "categorical"])
            # Guess type
            if isinstance(p_val, int): p_type.setCurrentText("int")
            elif isinstance(p_val, float): p_type.setCurrentText("float")
            else: p_type.setCurrentText("categorical")
            row.addWidget(p_type)
            
            self.param_layout.addLayout(row)
            self.param_checks[p_name] = (cb, p_type)
            
        self.param_scroll.setVisible(True)

    def get_model_data(self):
        params = {}
        for p_name, (cb, p_type) in self.param_checks.items():
            if cb.isChecked():
                t = p_type.currentText()
                if t == 'int': params[p_name] = {"enabled": True, "bits": 8, "type": "int", "min": 0, "max": 255}
                elif t == 'float': params[p_name] = {"enabled": True, "type": "float", "bits_mantissa": 3, "bits_exponent": 3, "bits_sign": 1}
                else: params[p_name] = {"enabled": True, "bits": 2, "type": "categorical", "values": ["Default"]}
        
        return {
            "name": self.name_edit.text().strip(),
            "import_path": self.path_edit.text().strip(),
            "params": params
        }

class IndividualConfigDialog(QDialog):
# ... (rest of the class)
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Individual Configuration")
        self.resize(900, 600)
        self.config = copy.deepcopy(current_config)
        self.model_widgets = {} # model_name -> {widgets}
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top Instructions
        header = QLabel("Select models to include in the evolution and configure their hyperparameters.")
        header.setStyleSheet("font-weight: bold; color: #f2cdcd;")
        main_layout.addWidget(header)

        # Splitter for Master-Detail
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left: Model List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.model_list = QListWidget()
        self.model_list.setSpacing(2)
        self.model_list.currentRowChanged.connect(self._on_model_changed)
        
        left_layout.addWidget(QLabel("Available Models:"))
        left_layout.addWidget(self.model_list)
        self.splitter.addWidget(left_widget)

        # Right: Config Stack
        self.stack = QStackedWidget()
        right_container = QGroupBox("Model Parameters")
        right_layout = QVBoxLayout(right_container)
        right_layout.addWidget(self.stack)
        self.splitter.addWidget(right_container)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([220, 680])
        main_layout.addWidget(self.splitter)

        # Populate
        for model in self.config['models']:
            self._add_model_to_ui(model)

        # Footer Buttons
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load .evoind")
        load_btn.clicked.connect(self._on_load)
        save_btn = QPushButton("Save .evoind")
        save_btn.clicked.connect(self._on_save)
        
        add_custom_btn = QPushButton("Add Custom Model...")
        add_custom_btn.clicked.connect(self._on_add_custom_model)
        
        apply_btn = QPushButton("Apply Configuration")
        apply_btn.setStyleSheet("background-color: #7fbf7f; color: #11151b; font-weight: bold;")
        apply_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(add_custom_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

        if self.model_list.count() > 0:
            self.model_list.setCurrentRow(0)

    def _on_add_custom_model(self):
        dialog = NewModelDialog(self)
        if dialog.exec():
            new_model = dialog.get_model_data()
            # 1. Update user_models.json
            user_models = []
            if os.path.exists(USER_MODELS_PATH):
                try:
                    with open(USER_MODELS_PATH, 'r') as f: user_models = json.load(f)
                except: pass
            
            user_models.append(new_model)
            with open(USER_MODELS_PATH, 'w') as f: json.dump(user_models, f, indent=4)
            
            # 2. Reload registry
            reload_registry()
            
            # 3. Update local config and UI
            from evo.models_registry import MODELS_REGISTRY
            self.config['models'] = copy.deepcopy(MODELS_REGISTRY)
            
            self.model_list.clear()
            while self.stack.count():
                w = self.stack.takeAt(0).widget()
                if w: w.deleteLater()
            self.model_widgets.clear()
            
            for model in self.config['models']:
                self._add_model_to_ui(model)
            self.model_list.setCurrentRow(self.model_list.count()-1)
            QMessageBox.information(self, "Success", f"Model '{new_model['name']}' added to registry.")

    def _add_model_to_ui(self, model):
        # 1. Add to List with Checkbox
        item = QListWidgetItem(model['name'])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if model.get('enabled', True) else Qt.Unchecked)
        self.model_list.addItem(item)

        # 2. Create Config Page
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(5, 5, 5, 5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        param_widgets = {}
        for p_name, p_data in model['params'].items():
            p_group = QGroupBox(p_name)
            p_group.setStyleSheet("QGroupBox { border: 1px solid #3b4757; margin-top: 10px; padding-top: 5px; }")
            p_form = QFormLayout()
            p_form.setVerticalSpacing(4)
            p_form.setLabelAlignment(Qt.AlignRight)
            
            p_enable = QCheckBox("Optimize this parameter")
            p_enable.setChecked(p_data.get('enabled', True))
            p_form.addRow(p_enable)

            p_type = QComboBox()
            p_type.addItems(["int", "categorical", "float"])
            p_type.setCurrentText(p_data['type'])
            p_type.setEnabled(p_enable.isChecked())
            p_form.addRow("Encoding:", p_type)

            type_container = QWidget()
            type_layout = QFormLayout(type_container)
            type_layout.setContentsMargins(0, 0, 0, 0)
            type_layout.setVerticalSpacing(4)
            type_container.setEnabled(p_enable.isChecked())
            p_form.addRow(type_container)

            widgets = {'enable': p_enable, 'type': p_type, 'container': type_container, 'layout': type_layout}
            param_widgets[p_name] = widgets

            def update_type_ui(idx, w=widgets, data=p_data):
                while w['layout'].count():
                    child = w['layout'].takeAt(0)
                    if child.widget(): child.widget().deleteLater()
                
                t = w['type'].currentText()
                if t == 'int':
                    bits = QSpinBox(); bits.setRange(1, 32); bits.setValue(data.get('bits', 8))
                    w['layout'].addRow("Bits:", bits); w['bits'] = bits
                    min_val = QSpinBox(); min_val.setRange(-999999, 999999); min_val.setValue(data.get('min', 0))
                    w['layout'].addRow("Min Value:", min_val); w['min'] = min_val
                elif t == 'categorical':
                    bits = QSpinBox(); bits.setRange(1, 32); bits.setValue(data.get('bits', 2))
                    w['layout'].addRow("Bits:", bits); w['bits'] = bits
                    vals = data.get('values', ["?"])
                    vals_lbl = QLabel(", ".join(vals)); vals_lbl.setWordWrap(True)
                    vals_lbl.setStyleSheet("color: #aab4c3; font-size: 8pt;")
                    w['layout'].addRow("Values:", vals_lbl)
                elif t == 'float':
                    m_bits = QSpinBox(); m_bits.setRange(1, 32); m_bits.setValue(data.get('bits_mantissa', 3))
                    w['layout'].addRow("Mantissa Bits:", m_bits); w['bits_mantissa'] = m_bits
                    e_bits = QSpinBox(); e_bits.setRange(1, 32); e_bits.setValue(data.get('bits_exponent', 3))
                    w['layout'].addRow("Exponent Bits:", e_bits); w['bits_exponent'] = e_bits
                    s_bits = QSpinBox(); s_bits.setRange(0, 1); s_bits.setValue(data.get('bits_sign', 1))
                    w['layout'].addRow("Sign Bit:", s_bits); w['bits_sign'] = s_bits

            p_type.currentIndexChanged.connect(lambda idx, w=widgets, d=p_data: update_type_ui(idx, w, d))
            p_enable.toggled.connect(p_type.setEnabled)
            p_enable.toggled.connect(type_container.setEnabled)
            
            update_type_ui(0, widgets, p_data)
            scroll_layout.addWidget(p_group)
            p_group.setLayout(p_form)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        page_layout.addWidget(scroll)
        
        self.stack.addWidget(page)
        self.model_widgets[model['name']] = {'item': item, 'params': param_widgets}

    def _on_model_changed(self, index):
        if index >= 0:
            self.stack.setCurrentIndex(index)

    def get_updated_config(self):
        updated_config = {'models': []}
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            m_name = item.text()
            m_data = next(m for m in self.config['models'] if m['name'] == m_name)
            
            model_entry = copy.deepcopy(m_data)
            model_entry['enabled'] = (item.checkState() == Qt.Checked)
            
            widgets_info = self.model_widgets[m_name]
            for p_name, p_data in model_entry['params'].items():
                w = widgets_info['params'][p_name]
                p_data['enabled'] = w['enable'].isChecked()
                p_data['type'] = w['type'].currentText()
                
                if p_data['type'] == 'int':
                    p_data['bits'] = w['bits'].value()
                    p_data['min'] = w['min'].value()
                elif p_data['type'] == 'categorical':
                    p_data['bits'] = w['bits'].value()
                elif p_data['type'] == 'float':
                    p_data['bits_mantissa'] = w['bits_mantissa'].value()
                    p_data['bits_exponent'] = w['bits_exponent'].value()
                    p_data['bits_sign'] = w['bits_sign'].value()
            
            updated_config['models'].append(model_entry)
        return updated_config

    def _on_save(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Individual Config", "", "Evo Individual Config (*.evoind)")
        if file_path:
            if not file_path.endswith(".evoind"): file_path += ".evoind"
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.get_updated_config(), f, indent=4)
                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Individual Config", "", "Evo Individual Config (*.evoind)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
                self.config = new_config
                # Rebuild UI
                self.model_list.clear()
                while self.stack.count():
                    widget = self.stack.takeAt(0).widget()
                    if widget: widget.deleteLater()
                self.model_widgets.clear()
                for model in self.config['models']:
                    self._add_model_to_ui(model)
                if self.model_list.count() > 0:
                    self.model_list.setCurrentRow(0)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

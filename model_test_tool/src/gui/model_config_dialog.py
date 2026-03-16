from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QLineEdit, QDoubleSpinBox, QPushButton, QMessageBox,
                             QGroupBox, QCheckBox, QSpinBox)
from PySide6.QtCore import Qt

class ModelConfigDialog(QDialog):
    def __init__(self, model_config, parent=None):
        super().__init__(parent)
        try:
            self.model_config = model_config
            self.setWindowTitle(f"配置模型 - {model_config.model_name}")
            self.setModal(True)
            self.resize(500, 400)
            
            self.init_ui()
            self.load_config()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create form layout for config fields
        form_layout = QFormLayout()
        
        # API Base
        self.api_base_edit = QLineEdit()
        form_layout.addRow("API Base:", self.api_base_edit)
        
        # API Key
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        form_layout.addRow("API Key:", self.api_key_edit)
        
        # Request Model (实际请求的模型名称)
        self.request_model_edit = QLineEdit()
        form_layout.addRow("请求模型名称:", self.request_model_edit)
        
        # System Message
        self.system_message_edit = QLineEdit()
        form_layout.addRow("System Message:", self.system_message_edit)
        
        # Temperature
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setRange(0.0, 2.0)
        self.temperature_spinbox.setSingleStep(0.1)
        form_layout.addRow("Temperature:", self.temperature_spinbox)
        
        # Max Tokens
        self.max_tokens_spinbox = QDoubleSpinBox()
        self.max_tokens_spinbox.setRange(1, 32768)
        self.max_tokens_spinbox.setDecimals(0)
        form_layout.addRow("Max Tokens:", self.max_tokens_spinbox)
        
        # Top P
        self.top_p_spinbox = QDoubleSpinBox()
        self.top_p_spinbox.setRange(0.0, 1.0)
        self.top_p_spinbox.setSingleStep(0.1)
        form_layout.addRow("Top P:", self.top_p_spinbox)
        
        # Frequency Penalty
        self.frequency_penalty_spinbox = QDoubleSpinBox()
        self.frequency_penalty_spinbox.setRange(-2.0, 2.0)
        self.frequency_penalty_spinbox.setSingleStep(0.1)
        form_layout.addRow("Frequency Penalty:", self.frequency_penalty_spinbox)
        
        # Presence Penalty
        self.presence_penalty_spinbox = QDoubleSpinBox()
        self.presence_penalty_spinbox.setRange(-2.0, 2.0)
        self.presence_penalty_spinbox.setSingleStep(0.1)
        form_layout.addRow("Presence Penalty:", self.presence_penalty_spinbox)
        
        # 请求超时设置
        self.timeout_spinbox = QDoubleSpinBox()
        self.timeout_spinbox.setRange(1.0, 600.0)  # 1秒到600秒（10分钟）
        self.timeout_spinbox.setSingleStep(1.0)
        self.timeout_spinbox.setDecimals(0)
        form_layout.addRow("请求超时(秒):", self.timeout_spinbox)
        
        # 并发请求设置
        self.concurrent_group = QGroupBox("并发请求设置")
        concurrent_layout = QFormLayout()
        
        # 是否开启并发
        self.enable_concurrent_checkbox = QCheckBox("开启并发请求")
        concurrent_layout.addRow("并发功能:", self.enable_concurrent_checkbox)
        
        # 最大并发请求数
        self.max_concurrent_spinbox = QSpinBox()
        self.max_concurrent_spinbox.setRange(1, 1000)
        self.max_concurrent_spinbox.setValue(10)
        concurrent_layout.addRow("每分钟最大并发数:", self.max_concurrent_spinbox)
        
        # 同时并发总数限制
        self.max_concurrent_total_spinbox = QSpinBox()
        self.max_concurrent_total_spinbox.setRange(-1, 10000)
        self.max_concurrent_total_spinbox.setValue(-1)
        self.max_concurrent_total_spinbox.setSpecialValueText("不限制")
        concurrent_layout.addRow("同时并发总数限制:", self.max_concurrent_total_spinbox)
        
        self.concurrent_group.setLayout(concurrent_layout)
        layout.addWidget(self.concurrent_group)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def load_config(self):
        self.api_base_edit.setText(self.model_config.api_base)
        self.api_key_edit.setText(self.model_config.api_key)
        self.request_model_edit.setText(self.model_config.request_model)  # 加载请求model
        self.system_message_edit.setText(self.model_config.system_message)  # 加载System Message
        self.temperature_spinbox.setValue(self.model_config.temperature)
        self.max_tokens_spinbox.setValue(self.model_config.max_tokens)
        self.top_p_spinbox.setValue(self.model_config.top_p)
        self.frequency_penalty_spinbox.setValue(self.model_config.frequency_penalty)
        self.presence_penalty_spinbox.setValue(self.model_config.presence_penalty)
        # 加载并发设置
        self.enable_concurrent_checkbox.setChecked(self.model_config.enable_concurrent)
        self.max_concurrent_spinbox.setValue(self.model_config.max_concurrent_requests)
        self.max_concurrent_total_spinbox.setValue(self.model_config.max_concurrent_total)
        # 加载超时设置
        self.timeout_spinbox.setValue(self.model_config.timeout)
        
    def save_config(self):
        # Validate inputs
        if not self.api_base_edit.text():
            QMessageBox.warning(self, "警告", "API Base 不能为空")
            return
            
        # 移除API Key不能为空的验证，允许为空
        # if not self.api_key_edit.text():
        #     QMessageBox.warning(self, "警告", "API Key 不能为空")
        #     return
            
        
        # Update model config
        self.model_config.api_base = self.api_base_edit.text()
        self.model_config.api_key = self.api_key_edit.text()  # 允许为空
        self.model_config.request_model = self.request_model_edit.text() or self.model_config.model_name  # 保存请求model
        self.model_config.system_message = self.system_message_edit.text()  # 保存System Message
        self.model_config.temperature = self.temperature_spinbox.value()
        self.model_config.max_tokens = int(self.max_tokens_spinbox.value())
        self.model_config.top_p = self.top_p_spinbox.value()
        self.model_config.frequency_penalty = self.frequency_penalty_spinbox.value()
        self.model_config.presence_penalty = self.presence_penalty_spinbox.value()
        # 保存并发设置
        self.model_config.enable_concurrent = self.enable_concurrent_checkbox.isChecked()
        self.model_config.max_concurrent_requests = self.max_concurrent_spinbox.value()
        self.model_config.max_concurrent_total = self.max_concurrent_total_spinbox.value()
        # 保存超时设置
        self.model_config.timeout = self.timeout_spinbox.value()
        
        self.accept()

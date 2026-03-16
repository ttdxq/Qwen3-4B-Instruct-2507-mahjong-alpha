import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QGroupBox,
    QCheckBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QTabWidget,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QInputDialog,
    QTableWidgetItem,
    QMenu,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal
from core.evaluator import ModelEvaluator
from core.model_config import ModelConfig
from gui.model_config_dialog import ModelConfigDialog
import json
import datetime
import threading


class EvaluationThread(QThread):
    progress_updated = Signal(int, int)
    result_updated = Signal(dict)
    evaluation_finished = Signal(dict)

    def __init__(
        self,
        evaluator,
        datasets,
        models,
        config,
        max_requests=None,
        model_max_requests=None,
        task_name=None,
        total_requests=None,
        previous_completed=0,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.datasets = datasets
        self.models = models
        self.config = config
        self.max_requests = max_requests  # 添加最大请求数限制参数
        self.model_max_requests = model_max_requests  # 添加每个模型的最大请求数限制参数
        self.task_name = task_name  # 添加任务名称参数
        # 新增保存参数
        self.total_requests = total_requests
        self.previous_completed = previous_completed
        # 添加停止标志
        self._stop_requested = False
        self._stop_lock = threading.Lock()

    def request_stop(self):
        """请求停止评估"""
        with self._stop_lock:
            self._stop_requested = True
            # 设置评估器的停止标志
            if hasattr(self.evaluator, "set_stop_flag"):
                self.evaluator.set_stop_flag(True)

    def is_stop_requested(self):
        """检查是否请求停止"""
        with self._stop_lock:
            return self._stop_requested

    def run(self):
        try:
            # 将任务名称添加到config中
            if self.task_name:
                self.config["task_name"] = self.task_name

            if self.max_requests or self.model_max_requests:
                # [修改] 调用 evaluate_with_progress_tracking 时传递全局参数
                results = self.evaluator.evaluate_with_progress_tracking(
                    self.datasets,
                    self.models,
                    self.config,
                    self.progress_updated,
                    self.result_updated,
                    self.max_requests,
                    self.model_max_requests,
                    total_task_requests=self.total_requests,  # 传参
                    previous_task_completed=self.previous_completed,  # 传参
                )
            else:
                # 使用原有的评估方法
                results = self.evaluator.evaluate(
                    self.datasets,
                    self.models,
                    self.config,
                    self.progress_updated,
                    self.result_updated,
                )

            # 检查是否请求停止
            if self.is_stop_requested():
                # 如果是停止请求导致的结束，仍需保存部分结果
                self.evaluation_finished.emit(results)
            else:
                self.evaluation_finished.emit(results)

        except Exception as e:
            # Handle exception
            self.evaluation_finished.emit({"error": str(e)})


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("大模型评测工具")
        
        # 设置默认窗口大小，稍后在showEvent中调整为屏幕尺寸的百分比
        self.resize(1200, 800)
        
        # 设置最小窗口大小
        self.setMinimumSize(1024, 768)

        # Initialize variables
        self.datasets = []
        self.models = []
        self.model_configs = {}
        self.output_dir = ""
        self.current_task_config = None  # 添加当前任务配置变量
        self.completed_tasks = set()  # 添加已完成任务集合
        self._task_evaluation_state = "Idle"
        self._stop_in_progress = False

        # Initialize UI first
        self.init_ui()

        # Load configuration if exists (after UI is created)
        self.config_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        self.load_config()

        # 从通用设置初始化评测配置的默认值
        self.init_task_config_from_general()

        # 重新填充列表以显示加载的数据
        self.populate_lists()

        # 程序启动时自动刷新任务文件列表
        self.refresh_task_files()

    def showEvent(self, event):
        """在窗口显示时调整大小以适配屏幕"""
        super().showEvent(event)
        # 确保在显示时调整窗口大小以适配屏幕
        try:
            screen_geometry = self.screen().size() if self.screen() else QApplication.primaryScreen().size()
            width = int(screen_geometry.width() * 0.6)  # 屏幕宽度的60%
            height = int(screen_geometry.height() * 0.6)  # 屏幕高度的60%
            self.resize(width, height)
        except:
            # 如果获取屏幕尺寸失败，使用默认大小
            self.resize(1200, 800)

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        # 可以在这里添加响应式布局调整逻辑
        # 目前使用Qt的内置布局管理器自动处理
        
    def init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # Create tabs - 先创建配置标签页，确保通用设置控件先被创建
        self.create_config_tab(tab_widget)
        self.create_task_tab(tab_widget)  # 任务管理作为主要评测页
        self.create_dataset_tab(tab_widget)
        self.create_model_tab(tab_widget)

        # 调整标签页顺序：测评在第一页，设置在最后一页
        # 移除所有标签页
        while tab_widget.count() > 0:
            tab_widget.removeTab(0)

        # 重新按正确顺序添加标签页
        self.create_task_tab(tab_widget)  # 测评页面放在第一位
        self.create_dataset_tab(tab_widget)
        self.create_model_tab(tab_widget)
        self.create_config_tab(tab_widget)  # 设置页面放在最后

        # Status bar
        self.statusBar().showMessage("就绪")

        # Populate lists with loaded data
        self.populate_lists()

    def populate_lists(self):
        """填充数据集和模型列表"""
        # 清空现有列表
        self.dataset_list.clear()
        self.model_list.clear()
        if hasattr(self, "eval_dataset_list"):
            self.eval_dataset_list.clear()
        if hasattr(self, "eval_model_list"):
            self.eval_model_list.clear()

        # 填充数据集列表
        for dataset in self.datasets:
            item = QListWidgetItem(dataset)
            self.dataset_list.addItem(item)
            # 同时填充任务管理中的数据集列表
            if hasattr(self, "eval_dataset_list"):
                eval_item = QListWidgetItem(dataset)
                self.eval_dataset_list.addItem(eval_item)

        # 填充模型列表
        for model in self.models:
            item = QListWidgetItem(model)
            self.model_list.addItem(item)
            # 同时填充任务管理中的模型列表
            if hasattr(self, "eval_model_list"):
                eval_item = QListWidgetItem(model)
                self.eval_model_list.addItem(eval_item)

    def create_dataset_tab(self, tab_widget):
        dataset_widget = QWidget()
        layout = QVBoxLayout()

        # Dataset selection
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout()

        self.dataset_list = QListWidget()
        # 设置列表自适应大小
        self.dataset_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dataset_layout.addWidget(self.dataset_list)

        dataset_button_layout = QHBoxLayout()
        self.add_dataset_button = QPushButton("添加数据集")
        self.add_dataset_button.clicked.connect(self.add_dataset)
        dataset_button_layout.addWidget(self.add_dataset_button)

        self.remove_dataset_button = QPushButton("移除选中")
        self.remove_dataset_button.clicked.connect(self.remove_dataset)
        dataset_button_layout.addWidget(self.remove_dataset_button)

        dataset_layout.addLayout(dataset_button_layout)
        dataset_group.setLayout(dataset_layout)

        layout.addWidget(dataset_group)
        dataset_widget.setLayout(layout)
        tab_widget.addTab(dataset_widget, "数据集")

    def create_model_tab(self, tab_widget):
        model_widget = QWidget()
        layout = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()

        self.model_list = QListWidget()
        # 启用右键菜单
        self.model_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_list.customContextMenuRequested.connect(self.show_model_context_menu)
        # 设置列表自适应大小
        self.model_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        model_layout.addWidget(self.model_list)

        model_button_layout = QHBoxLayout()
        self.add_model_button = QPushButton("添加模型")
        self.add_model_button.clicked.connect(self.add_model)
        model_button_layout.addWidget(self.add_model_button)

        self.remove_model_button = QPushButton("移除选中")
        self.remove_model_button.clicked.connect(self.remove_model)
        model_button_layout.addWidget(self.remove_model_button)

        self.config_model_button = QPushButton("配置模型")
        self.config_model_button.clicked.connect(self.config_model)
        model_button_layout.addWidget(self.config_model_button)

        self.copy_model_button = QPushButton("复制模型配置")
        self.copy_model_button.clicked.connect(self.copy_model_config)
        model_button_layout.addWidget(self.copy_model_button)

        model_layout.addLayout(model_button_layout)
        model_group.setLayout(model_layout)

        layout.addWidget(model_group)
        model_widget.setLayout(layout)
        tab_widget.addTab(model_widget, "模型")

    def create_config_tab(self, tab_widget):
        config_widget = QWidget()
        layout = QVBoxLayout()

        # 通用设置组
        general_group = self.create_general_settings_group()
        # 设置通用设置组自适应大小
        general_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(general_group)

        # 输出配置
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout()

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("结果输出目录:"))
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_button = QPushButton("浏览")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.output_dir_button)
        output_layout.addLayout(output_dir_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # 填充顶部
        layout.addStretch()

        config_widget.setLayout(layout)
        tab_widget.addTab(config_widget, "设置")

    def create_general_settings_group(self):
        """创建通用设置组，包含过滤选项等"""
        general_group = QGroupBox("通用设置")
        general_layout = QVBoxLayout()

        # Sample count
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("抽取样本数量:"))
        self.sample_count_spinbox = QSpinBox()
        self.sample_count_spinbox.setRange(1, 10000)
        self.sample_count_spinbox.setValue(10)
        sample_layout.addWidget(self.sample_count_spinbox)
        sample_layout.addStretch()
        general_layout.addLayout(sample_layout)

        # Evaluation times
        eval_layout = QHBoxLayout()
        eval_layout.addWidget(QLabel("每个样本评测次数:"))
        self.eval_times_spinbox = QSpinBox()
        self.eval_times_spinbox.setRange(1, 100)
        self.eval_times_spinbox.setValue(1)
        eval_layout.addWidget(self.eval_times_spinbox)
        eval_layout.addStretch()
        general_layout.addLayout(eval_layout)

        # Scoring
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("正确得分:"))
        self.correct_score_spinbox = QDoubleSpinBox()
        self.correct_score_spinbox.setRange(0, 100)
        self.correct_score_spinbox.setValue(1.0)
        self.correct_score_spinbox.setSingleStep(0.1)
        score_layout.addWidget(self.correct_score_spinbox)

        score_layout.addWidget(QLabel("错误得分:"))
        self.wrong_score_spinbox = QDoubleSpinBox()
        self.wrong_score_spinbox.setRange(-100, 0)
        self.wrong_score_spinbox.setValue(0.0)
        self.wrong_score_spinbox.setSingleStep(0.1)
        score_layout.addWidget(self.wrong_score_spinbox)
        score_layout.addStretch()
        general_layout.addLayout(score_layout)

        # Score aggregation
        agg_layout = QHBoxLayout()
        agg_layout.addWidget(QLabel("同样本分数计算方式:"))
        self.score_agg_combo = QComboBox()
        self.score_agg_combo.addItem("累加", "sum")
        self.score_agg_combo.addItem("平均", "avg")
        agg_layout.addWidget(self.score_agg_combo)
        agg_layout.addStretch()
        general_layout.addLayout(agg_layout)

        # 添加过滤选项
        filter_layout = QHBoxLayout()
        self.filter_empty_results_checkbox = QCheckBox("保存结果时过滤空结果")
        self.filter_empty_results_checkbox.setToolTip(
            "勾选此项会在保存结果时过滤掉提取结果为空的请求"
        )
        self.filter_empty_results_checkbox.setChecked(True)  # 默认勾选
        filter_layout.addWidget(self.filter_empty_results_checkbox)
        filter_layout.addStretch()
        general_layout.addLayout(filter_layout)

        # Turn ratios configuration
        turn_ratio_group = QGroupBox("巡目比例配置")
        turn_ratio_layout = QVBoxLayout()

        # 请求超时设置
        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(QLabel("请求超时(秒):"))
        self.timeout_spinbox = QDoubleSpinBox()
        self.timeout_spinbox.setRange(1.0, 600.0)
        self.timeout_spinbox.setSingleStep(1.0)
        self.timeout_spinbox.setDecimals(0)
        self.timeout_spinbox.setValue(60.0)
        timeout_layout.addWidget(self.timeout_spinbox)
        timeout_layout.addStretch()
        general_layout.addLayout(timeout_layout)

        # Early turn ratio
        early_layout = QHBoxLayout()
        early_layout.addWidget(QLabel("早巡比例:"))
        self.early_ratio_spinbox = QDoubleSpinBox()
        self.early_ratio_spinbox.setRange(0, 1)
        self.early_ratio_spinbox.setValue(0.33)
        self.early_ratio_spinbox.setSingleStep(0.01)
        early_layout.addWidget(self.early_ratio_spinbox)
        early_layout.addStretch()
        turn_ratio_layout.addLayout(early_layout)

        # Mid turn ratio
        mid_layout = QHBoxLayout()
        mid_layout.addWidget(QLabel("中巡比例:"))
        self.mid_ratio_spinbox = QDoubleSpinBox()
        self.mid_ratio_spinbox.setRange(0, 1)
        self.mid_ratio_spinbox.setValue(0.33)
        self.mid_ratio_spinbox.setSingleStep(0.01)
        mid_layout.addWidget(self.mid_ratio_spinbox)
        mid_layout.addStretch()
        turn_ratio_layout.addLayout(mid_layout)

        # Late turn ratio
        late_layout = QHBoxLayout()
        late_layout.addWidget(QLabel("晚巡比例:"))
        self.late_ratio_spinbox = QDoubleSpinBox()
        self.late_ratio_spinbox.setRange(0, 1)
        self.late_ratio_spinbox.setValue(0.34)
        self.late_ratio_spinbox.setSingleStep(0.01)
        late_layout.addWidget(self.late_ratio_spinbox)
        late_layout.addStretch()
        turn_ratio_layout.addLayout(late_layout)

        turn_ratio_group.setLayout(turn_ratio_layout)
        general_layout.addWidget(turn_ratio_group)

        # 连接信号，实现通用设置与评测配置的同步和自动保存
        self.sample_count_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.sample_count_spinbox.valueChanged.connect(self.auto_save_config)
        self.eval_times_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.eval_times_spinbox.valueChanged.connect(self.auto_save_config)
        self.correct_score_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.correct_score_spinbox.valueChanged.connect(self.auto_save_config)
        self.wrong_score_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.wrong_score_spinbox.valueChanged.connect(self.auto_save_config)
        self.score_agg_combo.currentIndexChanged.connect(self.sync_config_to_task)
        self.score_agg_combo.currentIndexChanged.connect(self.auto_save_config)
        self.filter_empty_results_checkbox.stateChanged.connect(self.auto_save_config)

        general_group.setLayout(general_layout)
        return general_group

        # Sample count
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("抽取样本数量:"))
        self.sample_count_spinbox = QSpinBox()
        self.sample_count_spinbox.setRange(1, 10000)
        self.sample_count_spinbox.setValue(10)
        sample_layout.addWidget(self.sample_count_spinbox)
        sample_layout.addStretch()
        general_layout.addLayout(sample_layout)

        # Evaluation times
        eval_layout = QHBoxLayout()
        eval_layout.addWidget(QLabel("每个样本评测次数:"))
        self.eval_times_spinbox = QSpinBox()
        self.eval_times_spinbox.setRange(1, 100)
        self.eval_times_spinbox.setValue(1)
        eval_layout.addWidget(self.eval_times_spinbox)
        eval_layout.addStretch()
        general_layout.addLayout(eval_layout)

        # Scoring
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("正确得分:"))
        self.correct_score_spinbox = QDoubleSpinBox()
        self.correct_score_spinbox.setRange(0, 100)
        self.correct_score_spinbox.setValue(1.0)
        self.correct_score_spinbox.setSingleStep(0.1)
        score_layout.addWidget(self.correct_score_spinbox)

        score_layout.addWidget(QLabel("错误得分:"))
        self.wrong_score_spinbox = QDoubleSpinBox()
        self.wrong_score_spinbox.setRange(-100, 0)
        self.wrong_score_spinbox.setValue(0.0)
        self.wrong_score_spinbox.setSingleStep(0.1)
        score_layout.addWidget(self.wrong_score_spinbox)
        score_layout.addStretch()
        general_layout.addLayout(score_layout)

        # Score aggregation
        agg_layout = QHBoxLayout()
        agg_layout.addWidget(QLabel("同样本分数计算方式:"))
        self.score_agg_combo = QComboBox()
        self.score_agg_combo.addItem("累加", "sum")
        self.score_agg_combo.addItem("平均", "avg")
        agg_layout.addWidget(self.score_agg_combo)
        agg_layout.addStretch()
        general_layout.addLayout(agg_layout)

        # 添加过滤选项
        filter_layout = QHBoxLayout()
        self.filter_empty_results_checkbox = QCheckBox("保存结果时过滤空结果")
        self.filter_empty_results_checkbox.setToolTip(
            "勾选此项会在保存结果时过滤掉提取结果为空的请求"
        )
        self.filter_empty_results_checkbox.setChecked(True)  # 默认勾选
        filter_layout.addWidget(self.filter_empty_results_checkbox)
        filter_layout.addStretch()
        general_layout.addLayout(filter_layout)

        # Turn ratios configuration
        turn_ratio_group = QGroupBox("巡目比例配置")
        turn_ratio_layout = QVBoxLayout()

        # Early turn ratio
        early_layout = QHBoxLayout()
        early_layout.addWidget(QLabel("早巡比例:"))
        self.early_ratio_spinbox = QDoubleSpinBox()
        self.early_ratio_spinbox.setRange(0, 1)
        self.early_ratio_spinbox.setValue(0.33)
        self.early_ratio_spinbox.setSingleStep(0.01)
        early_layout.addWidget(self.early_ratio_spinbox)
        early_layout.addStretch()
        turn_ratio_layout.addLayout(early_layout)

        # Mid turn ratio
        mid_layout = QHBoxLayout()
        mid_layout.addWidget(QLabel("中巡比例:"))
        self.mid_ratio_spinbox = QDoubleSpinBox()
        self.mid_ratio_spinbox.setRange(0, 1)
        self.mid_ratio_spinbox.setValue(0.33)
        self.mid_ratio_spinbox.setSingleStep(0.01)
        mid_layout.addWidget(self.mid_ratio_spinbox)
        mid_layout.addStretch()
        turn_ratio_layout.addLayout(mid_layout)

        # Late turn ratio
        late_layout = QHBoxLayout()
        late_layout.addWidget(QLabel("晚巡比例:"))
        self.late_ratio_spinbox = QDoubleSpinBox()
        self.late_ratio_spinbox.setRange(0, 1)
        self.late_ratio_spinbox.setValue(0.34)
        self.late_ratio_spinbox.setSingleStep(0.01)
        late_layout.addWidget(self.late_ratio_spinbox)
        late_layout.addStretch()
        turn_ratio_layout.addLayout(late_layout)

        turn_ratio_group.setLayout(turn_ratio_layout)
        general_layout.addWidget(turn_ratio_group)

        # 连接信号，实现通用设置与评测配置的同步和自动保存
        self.sample_count_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.sample_count_spinbox.valueChanged.connect(self.auto_save_config)
        self.eval_times_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.eval_times_spinbox.valueChanged.connect(self.auto_save_config)
        self.correct_score_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.correct_score_spinbox.valueChanged.connect(self.auto_save_config)
        self.wrong_score_spinbox.valueChanged.connect(self.sync_config_to_task)
        self.wrong_score_spinbox.valueChanged.connect(self.auto_save_config)
        self.score_agg_combo.currentIndexChanged.connect(self.sync_config_to_task)
        self.score_agg_combo.currentIndexChanged.connect(self.auto_save_config)
        self.filter_empty_results_checkbox.stateChanged.connect(self.auto_save_config)

        general_group.setLayout(general_layout)
        return general_group

    def sync_config_to_task(self):
        """同步通用设置到评测配置"""
        if hasattr(self, "task_sample_count_spinbox"):
            self.task_sample_count_spinbox.setValue(self.sample_count_spinbox.value())
        if hasattr(self, "task_eval_times_spinbox"):
            self.task_eval_times_spinbox.setValue(self.eval_times_spinbox.value())
        if hasattr(self, "task_correct_score_spinbox"):
            self.task_correct_score_spinbox.setValue(self.correct_score_spinbox.value())
        if hasattr(self, "task_wrong_score_spinbox"):
            self.task_wrong_score_spinbox.setValue(self.wrong_score_spinbox.value())
        if hasattr(self, "task_score_agg_combo"):
            self.task_score_agg_combo.setCurrentIndex(
                self.score_agg_combo.currentIndex()
            )
        if hasattr(self, "task_early_ratio_spinbox") and hasattr(
            self, "early_ratio_spinbox"
        ):
            self.task_early_ratio_spinbox.setValue(self.early_ratio_spinbox.value())
        if hasattr(self, "task_mid_ratio_spinbox") and hasattr(
            self, "mid_ratio_spinbox"
        ):
            self.task_mid_ratio_spinbox.setValue(self.mid_ratio_spinbox.value())
        if hasattr(self, "task_late_ratio_spinbox") and hasattr(
            self, "late_ratio_spinbox"
        ):
            self.task_late_ratio_spinbox.setValue(self.late_ratio_spinbox.value())

    def select_all_datasets(self):
        """全选数据集"""
        for i in range(self.eval_dataset_list.count()):
            item = self.eval_dataset_list.item(i)
            # Validate file exists before selection
            file_path = item.text()
            if file_path and not os.path.exists(file_path):
                QMessageBox.warning(self, "警告", f"数据集文件不存在: {file_path}")
                item.setSelected(False)
                continue  # Skip invalid files
            item.setSelected(True)

    def select_none_datasets(self):
        """清空数据集选择"""
        for i in range(self.eval_dataset_list.count()):
            item = self.eval_dataset_list.item(i)
            item.setSelected(False)

    def select_all_models(self):
        """全选模型"""
        for i in range(self.eval_model_list.count()):
            item = self.eval_model_list.item(i)
            item.setSelected(True)

    def select_none_models(self):
        """清空模型选择"""
        for i in range(self.eval_model_list.count()):
            item = self.eval_model_list.item(i)
            item.setSelected(False)

    def create_evaluation_task(self):
        """创建并保存测评任务"""
        # 检查输出目录是否已设置
        if not self.output_dir or not os.path.exists(self.output_dir):
            QMessageBox.warning(self, "警告", "请先设置有效的结果输出目录")
            return

        # 获取选中的数据集
        selected_datasets = []
        for i in range(self.eval_dataset_list.count()):
            item = self.eval_dataset_list.item(i)
            if item.isSelected():
                selected_datasets.append(item.text())

        if not selected_datasets:
            QMessageBox.warning(self, "警告", "请先选择至少一个数据集")
            return

        # 获取选中的模型
        selected_models = []
        for i in range(self.eval_model_list.count()):
            item = self.eval_model_list.item(i)
            if item.isSelected():
                selected_models.append(item.text())

        if not selected_models:
            QMessageBox.warning(self, "警告", "请先选择至少一个模型")
            return

        # 生成任务名称
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"task_{timestamp}"  # 生成任务名称

        # 构建任务配置
        task_config = {
            "task_name": task_name,  # 添加任务名称
            "datasets": selected_datasets,
            "models": selected_models,
            "config": {
                "sample_count": self.task_sample_count_spinbox.value(),
                "eval_times": self.task_eval_times_spinbox.value(),
                "correct_score": self.task_correct_score_spinbox.value(),
                "wrong_score": self.task_wrong_score_spinbox.value(),
                "score_agg": self.task_score_agg_combo.currentData(),
                "output_dir": self.output_dir,
                "filter_empty_results": self.filter_empty_results_checkbox.isChecked(),
                "turn_ratios": {
                    "early": self.task_early_ratio_spinbox.value(),
                    "mid": self.task_mid_ratio_spinbox.value(),
                    "late": self.task_late_ratio_spinbox.value(),
                },
            },
            "model_configs": {},
        }

        # 添加模型配置
        for model_name in selected_models:
            if model_name in self.model_configs:
                task_config["model_configs"][model_name] = self.model_configs[
                    model_name
                ].to_dict()

        # 从数据集中抽取样本并保存
        from core.evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        sampled_data = {}
        total_requests = 0  # 计算总请求数

        for dataset_path in selected_datasets:
            samples = evaluator.read_dataset(
                dataset_path, task_config["config"]["sample_count"]
            )
            sampled_data[dataset_path] = samples
            # 计算总请求数：样本数 * 每个样本评测次数 * 模型数
            total_requests += (
                len(samples)
                * task_config["config"]["eval_times"]
                * len(selected_models)
            )

        task_config["sampled_data"] = sampled_data
        task_config["total_requests"] = total_requests
        # 完成进度仅在内存和temp中维护，不写入任务文件
        task_config["completed_requests"] = 0
        task_config["completed_models"] = {}
        task_config["is_completed"] = False  # 添加任务完成状态

        # 为任务创建独立文件夹
        task_dir = os.path.join(self.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # 创建temp子文件夹用于存放临时数据
        temp_dir = os.path.join(task_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # 生成任务文件名
        task_filename = f"evaluation_task_{timestamp}.json"
        task_filepath = os.path.join(task_dir, task_filename)

        # 保存任务文件到任务文件夹（不写入完成进度字段）
        try:
            file_task_config = task_config.copy()
            file_task_config.pop("completed_requests", None)
            file_task_config.pop("completed_models", None)
            with open(task_filepath, "w", encoding="utf-8") as f:
                json.dump(file_task_config, f, ensure_ascii=False, indent=2)
            QMessageBox.information(
                self,
                "提示",
                f"测评任务已保存到: {task_filepath}\n总请求数: {total_requests}",
            )
            # 创建任务后自动刷新任务文件列表
            self.refresh_task_files()
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存测评任务失败: {str(e)}")

    def refresh_task_files(self):
        """刷新任务文件列表，隐藏已完成的任务（包括剩余请求数为零的任务）"""
        self.task_file_combo.clear()
        if os.path.exists(self.output_dir):
            # 首先检查输出目录根目录下的任务文件
            for filename in os.listdir(self.output_dir):
                if filename.startswith("evaluation_task_") and filename.endswith(
                    ".json"
                ):
                    # 检查任务是否已完成（包括剩余请求数为零）
                    task_filepath = os.path.join(self.output_dir, filename)
                    try:
                        with open(task_filepath, "r", encoding="utf-8") as f:
                            task_config = json.load(f)
                            # 读取temp中的完成数
                            task_dir = os.path.dirname(task_filepath)
                            temp_dir = os.path.join(task_dir, "temp")
                            completed_requests = 0
                            if os.path.exists(temp_dir):
                                import glob

                                temp_files = glob.glob(
                                    os.path.join(temp_dir, "temp_progress_*.json")
                                )
                                if temp_files:
                                    latest = sorted(temp_files)[-1]
                                    try:
                                        with open(latest, "r", encoding="utf-8") as tf:
                                            temp_data = json.load(tf)
                                            completed_requests = (
                                                temp_data.get("completed_requests", 0)
                                                or 0
                                            )
                                    except Exception:
                                        completed_requests = 0
                            # 检查任务是否已完成或剩余请求数为零
                            is_completed = task_config.get("is_completed", False)
                            total_requests = task_config.get("total_requests", 0)
                            remaining_requests = total_requests - completed_requests
                            has_zero_remaining_requests = remaining_requests <= 0

                            # 如果任务已完成或剩余请求数为零，不显示在列表中
                            if not is_completed and not has_zero_remaining_requests:
                                self.task_file_combo.addItem(filename)
                    except Exception:
                        # 如果无法读取任务文件，仍然显示它
                        self.task_file_combo.addItem(filename)

            # 然后检查每个任务文件夹中的任务文件
            for dirname in os.listdir(self.output_dir):
                dir_path = os.path.join(self.output_dir, dirname)
                if os.path.isdir(dir_path) and dirname.startswith("task_"):
                    for filename in os.listdir(dir_path):
                        if filename.startswith(
                            "evaluation_task_"
                        ) and filename.endswith(".json"):
                            task_filepath = os.path.join(dir_path, filename)
                            try:
                                with open(task_filepath, "r", encoding="utf-8") as f:
                                    task_config = json.load(f)
                                    # 检查任务是否已完成或剩余请求数为零
                                    is_completed = task_config.get(
                                        "is_completed", False
                                    )
                                    total_requests = task_config.get(
                                        "total_requests", 0
                                    )
                                    # 读取temp中的完成数
                                    temp_dir = os.path.join(dir_path, "temp")
                                    completed_requests = 0
                                    if os.path.exists(temp_dir):
                                        import glob

                                        temp_files = glob.glob(
                                            os.path.join(
                                                temp_dir, "temp_progress_*.json"
                                            )
                                        )
                                        if temp_files:
                                            latest = sorted(temp_files)[-1]
                                            try:
                                                with open(
                                                    latest, "r", encoding="utf-8"
                                                ) as tf:
                                                    temp_data = json.load(tf)
                                                    completed_requests = (
                                                        temp_data.get(
                                                            "completed_requests", 0
                                                        )
                                                        or 0
                                                    )
                                            except Exception:
                                                completed_requests = 0
                                    remaining_requests = (
                                        total_requests - completed_requests
                                    )
                                    has_zero_remaining_requests = (
                                        remaining_requests <= 0
                                    )

                                    # 如果任务已完成或剩余请求数为零，不显示在列表中
                                    if (
                                        not is_completed
                                        and not has_zero_remaining_requests
                                    ):
                                        # 显示相对路径，格式为 "task_folder/filename"
                                        relative_path = os.path.join(dirname, filename)
                                        self.task_file_combo.addItem(relative_path)
                            except Exception:
                                # 如果无法读取任务文件，仍然显示它
                                relative_path = os.path.join(dirname, filename)
                                self.task_file_combo.addItem(relative_path)

    def _check_zero_remaining_requests(self, task_config):
        """
        检查任务的剩余请求数是否为零
        剩余请求数 = 总请求数 - 已完成请求数
        """
        total_requests = task_config.get("total_requests", 0)
        completed_requests = task_config.get("completed_requests", 0)
        remaining_requests = total_requests - completed_requests
        return remaining_requests <= 0

    def _check_zero_remaining_requests(self, task_config):
        """
        检查任务的剩余请求数是否为零
        剩余请求数 = 总请求数 - 已完成请求数
        """
        total_requests = task_config.get("total_requests", 0)
        completed_requests = task_config.get("completed_requests", 0)
        remaining_requests = total_requests - completed_requests
        return remaining_requests <= 0

    def load_evaluation_task(self):
        """加载测评任务 - 选择后自动加载"""
        task_filename = self.task_file_combo.currentText()
        if not task_filename:
            return  # 如果没有选择任务，直接返回，不显示警告

        # 检查任务文件路径，支持根目录和任务文件夹中的文件
        task_filepath = os.path.join(self.output_dir, task_filename)
        # 如果直接在根目录找不到，尝试在任务文件夹中查找
        if not os.path.exists(task_filepath):
            # 检查是否为 "task_folder/filename" 格式
            if "/" in task_filename or os.sep in task_filename:
                parts = task_filename.split(os.sep if os.sep in task_filename else "/")
                if len(parts) == 2:
                    task_folder, filename = parts
                    task_filepath = os.path.join(self.output_dir, task_folder, filename)
            else:
                # 如果是简单文件名，直接在根目录查找
                task_filepath = os.path.join(self.output_dir, task_filename)

        if not os.path.exists(task_filepath):
            QMessageBox.warning(self, "警告", "任务文件不存在")
            return

        try:
            with open(task_filepath, "r", encoding="utf-8") as f:
                task_config = json.load(f)

            # 从temp文件读取最新进度（如果存在），否则视为0
            task_dir = os.path.dirname(task_filepath)
            temp_dir = os.path.join(task_dir, "temp")
            completed_from_temp = 0
            model_completed_from_temp = {}
            if os.path.exists(temp_dir):
                import glob

                temp_files = glob.glob(os.path.join(temp_dir, "temp_progress_*.json"))
                if temp_files:
                    latest = sorted(temp_files)[-1]
                    try:
                        with open(latest, "r", encoding="utf-8") as tf:
                            temp_data = json.load(tf)
                            completed_from_temp = (
                                temp_data.get("completed_requests", 0) or 0
                            )
                            model_completed_from_temp = (
                                temp_data.get("model_completed_requests", {}) or {}
                            )
                    except Exception:
                        completed_from_temp = 0
                        model_completed_from_temp = {}

            # 在内存中补齐完成进度，但不写回任务文件
            task_config["completed_requests"] = completed_from_temp
            task_config["completed_models"] = (
                task_config.get("completed_models", {}) or {}
            )

            # 从 temp_progress 补齐按模型的完成请求数（用于刷新后正确显示剩余请求数）
            if (
                isinstance(model_completed_from_temp, dict)
                and task_config.get("completed_models") is not None
            ):
                try:
                    total_requests_per_model = (
                        len(task_config.get("datasets", []))
                        * int(task_config.get("config", {}).get("sample_count", 0) or 0)
                        * int(task_config.get("config", {}).get("eval_times", 0) or 0)
                    )
                except Exception:
                    total_requests_per_model = 0

                for model_name, completed_count in model_completed_from_temp.items():
                    # 只记录任务内的模型，避免 temp 里混入其他任务数据
                    if model_name not in (task_config.get("models") or []):
                        continue
                    if model_name not in task_config["completed_models"]:
                        task_config["completed_models"][model_name] = {}
                    try:
                        task_config["completed_models"][model_name][
                            "completed_requests"
                        ] = int(completed_count or 0)
                    except Exception:
                        task_config["completed_models"][model_name][
                            "completed_requests"
                        ] = 0
                    if total_requests_per_model > 0:
                        task_config["completed_models"][model_name][
                            "total_requests"
                        ] = total_requests_per_model

            # 保存任务配置到实例变量
            self.current_task_config = task_config

            # 显示任务详情
            details = f"数据集: {', '.join(task_config['datasets'])}\n"
            details += f"模型: {', '.join(task_config['models'])}\n"
            details += f"样本数: {task_config['config']['sample_count']}\n"
            details += f"每个样本评测次数: {task_config['config']['eval_times']}\n"
            details += f"总请求数: {task_config['total_requests']}\n"
            details += f"已完成请求数: {task_config.get('completed_requests', 0)}\n"
            details += (
                f"已完成模型: {list(task_config.get('completed_models', {}).keys())}"
            )
            self.task_details_text.setPlainText(details)

            # 填充模型请求数表格
            self.update_model_requests_table(task_config["models"])

        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载测评任务失败: {str(e)}")

    def update_model_requests_table(self, models):
        """更新模型请求数表格"""
        self.model_requests_table.setRowCount(len(models))
        for i, model in enumerate(models):
            # 选择复选框
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # 默认选中
            checkbox.stateChanged.connect(self.on_model_checkbox_changed)
            self.model_requests_table.setCellWidget(i, 0, checkbox)

            # 模型名称
            model_item = QTableWidgetItem(model)
            model_item.setFlags(
                model_item.flags() & ~Qt.ItemIsEditable
            )  # 设置为不可编辑
            self.model_requests_table.setItem(i, 1, model_item)

            # 请求数设置 - 使用SpinBox
            from PySide6.QtWidgets import QSpinBox, QAbstractItemView

            spinbox = QSpinBox()
            spinbox.setRange(0, 10000)  # 允许0，请求数为0表示不使用该模型
            spinbox.setValue(10)  # 默认值
            # 移除连接到不存在的方法的信号
            # spinbox.valueChanged.connect(self.on_request_count_changed)

            # 计算每个模型的剩余请求数
            if hasattr(self, "current_task_config"):
                # 从任务配置中获取该模型的总请求数
                # 如果completed_models中有记录，使用其中的total_requests
                # 否则，计算默认值：数据集数量 * 样本数 * 每个样本评测次数
                if model in self.current_task_config.get("completed_models", {}):
                    total_requests_per_model = self.current_task_config[
                        "completed_models"
                    ][model].get(
                        "total_requests",
                        len(self.current_task_config["datasets"])
                        * self.current_task_config["config"]["sample_count"]
                        * self.current_task_config["config"]["eval_times"],
                    )
                else:
                    total_requests_per_model = (
                        len(self.current_task_config["datasets"])
                        * self.current_task_config["config"]["sample_count"]
                        * self.current_task_config["config"]["eval_times"]
                    )
                # 获取该模型已完成的请求数
                completed_for_model = (
                    self.current_task_config.get("completed_models", {})
                    .get(model, {})
                    .get("completed_requests", 0)
                )
                # 计算剩余请求数：总请求数 - 已完成请求数
                remaining_requests = total_requests_per_model - completed_for_model
                if remaining_requests > 0:
                    spinbox.setValue(
                        min(remaining_requests, 100)
                    )  # 默认设置为剩余请求数或100，取较小值
                else:
                    # 如果剩余请求数为0或负数，禁用复选框和SpinBox
                    checkbox.setChecked(False)
                    checkbox.setEnabled(False)
                    spinbox.setValue(0)
                    spinbox.setEnabled(False)
            self.model_requests_table.setCellWidget(i, 2, spinbox)

            # 添加剩余请求数显示 - 第4列
            from PySide6.QtWidgets import QLabel

            remaining_label = QLabel()
            if hasattr(self, "current_task_config"):
                # 从任务配置中获取该模型的总请求数
                if model in self.current_task_config.get("completed_models", {}):
                    total_requests_per_model = self.current_task_config[
                        "completed_models"
                    ][model].get(
                        "total_requests",
                        len(self.current_task_config["datasets"])
                        * self.current_task_config["config"]["sample_count"]
                        * self.current_task_config["config"]["eval_times"],
                    )
                else:
                    total_requests_per_model = (
                        len(self.current_task_config["datasets"])
                        * self.current_task_config["config"]["sample_count"]
                        * self.current_task_config["config"]["eval_times"]
                    )
                # 获取该模型已完成的请求数
                completed_for_model = (
                    self.current_task_config.get("completed_models", {})
                    .get(model, {})
                    .get("completed_requests", 0)
                )
                # 计算剩余请求数：每个模型的总请求数减去已完成的请求数
                remaining_requests = total_requests_per_model - completed_for_model
                remaining_label.setText(str(remaining_requests))
                # 如果剩余请求数为0或负数，设置灰色样式
                if remaining_requests <= 0:
                    remaining_label.setStyleSheet("color: gray;")
                else:
                    remaining_label.setStyleSheet("")
            else:
                remaining_label.setText("0")
                remaining_label.setStyleSheet("color: gray;")
            self.model_requests_table.setCellWidget(i, 3, remaining_label)

    def on_model_checkbox_changed(self, state):
        """模型复选框状态改变时的处理"""
        sender = self.sender()
        if sender:
            # 找到发送信号的复选框所在的行
            for row in range(self.model_requests_table.rowCount()):
                checkbox = self.model_requests_table.cellWidget(row, 0)
                if checkbox == sender:
                    spinbox = self.model_requests_table.cellWidget(row, 2)
                    if spinbox:
                        if state == Qt.Unchecked:
                            spinbox.setValue(0)  # 取消选中时请求数设为0
                        else:
                            # 重新选中时检查剩余请求数
                            model_item = self.model_requests_table.item(row, 1)
                            if model_item and hasattr(self, "current_task_config"):
                                model = model_item.text()
                                total_requests_per_model = (
                                    len(self.current_task_config["datasets"])
                                    * self.current_task_config["config"]["sample_count"]
                                    * self.current_task_config["config"]["eval_times"]
                                )
                                completed_for_model = (
                                    self.current_task_config.get("completed_models", {})
                                    .get(model, {})
                                    .get("completed_requests", 0)
                                )
                                remaining_requests = (
                                    total_requests_per_model - completed_for_model
                                )
                                if remaining_requests > 0:
                                    # 如果还有剩余请求数，设置默认值
                                    spinbox.setValue(
                                        min(remaining_requests, 10)
                                    )  # 设置为剩余请求数或10，取较小值
                                    spinbox.setEnabled(True)
                                else:
                                    # 如果没有剩余请求数，保持为0并禁用
                                    spinbox.setValue(0)
                                    spinbox.setEnabled(False)
                                    checkbox.setChecked(False)  # 重新禁用复选框
                            else:
                                spinbox.setValue(10)  # 默认值
                    break

    def update_remaining_requests_display(self):
        """更新剩余请求数显示"""
        if not hasattr(self, "current_task_config"):
            return

        for i in range(self.model_requests_table.rowCount()):
            model_item = self.model_requests_table.item(i, 1)  # 模型名称在第1列
            if model_item:
                model = model_item.text()
                remaining_label = self.model_requests_table.cellWidget(
                    i, 3
                )  # 剩余请求数在第3列
                if remaining_label:
                    total_requests_per_model = (
                        len(self.current_task_config["datasets"])
                        * self.current_task_config["config"]["sample_count"]
                        * self.current_task_config["config"]["eval_times"]
                    )
                    completed_for_model = (
                        self.current_task_config.get("completed_models", {})
                        .get(model, {})
                        .get("completed_requests", 0)
                    )
                    remaining_requests = total_requests_per_model - completed_for_model
                    remaining_label.setText(str(remaining_requests))

    def select_all_task_models(self):
        """全选任务模型"""
        # 这里应该是选择模型请求数表格中的模型
        for i in range(self.model_requests_table.rowCount()):
            checkbox = self.model_requests_table.cellWidget(i, 0)  # 选择复选框在第0列
            if checkbox:
                checkbox.setChecked(True)

    def select_none_task_models(self):
        """清空任务模型选择"""
        # 这里应该是取消选择模型请求数表格中的模型
        for i in range(self.model_requests_table.rowCount()):
            checkbox = self.model_requests_table.cellWidget(i, 0)  # 选择复选框在第0列
            if checkbox:
                checkbox.setChecked(False)

    def start_task_evaluation(self):
        """开始执行测评任务"""
        if self._task_evaluation_state in ("Running", "Stopping"):
            return

        if not hasattr(self, "current_task_config") or not self.current_task_config:
            QMessageBox.warning(self, "警告", "请先加载一个测评任务")
            return

        # 获取选中的模型和请求数
        selected_models = {}
        for i in range(self.model_requests_table.rowCount()):
            checkbox = self.model_requests_table.cellWidget(i, 0)
            model_item = self.model_requests_table.item(i, 1)
            spinbox = self.model_requests_table.cellWidget(i, 2)

            if checkbox and checkbox.isChecked() and model_item and spinbox:
                model_name = model_item.text()
                request_count = spinbox.value()
                if request_count > 0:
                    selected_models[model_name] = request_count

        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型并设置请求数大于0")
            return

        # 创建评测器
        from core.evaluator import ModelEvaluator

        evaluator = ModelEvaluator()

        # 从任务配置中获取数据集和配置
        datasets = self.current_task_config["datasets"]
        config = self.current_task_config["config"].copy()
        config["model_configs"] = self.current_task_config.get("model_configs", {})
        # 使用任务创建时保存的抽样数据，确保多次运行/中断续跑时 sample_id 一致
        if isinstance(self.current_task_config.get("sampled_data"), dict):
            config["preloaded_samples"] = self.current_task_config.get(
                "sampled_data", {}
            )

        # [修改前]
        # total_task_requests = self.current_task_config.get("total_requests", 0)
        # previous_completed_requests = self.current_task_config.get("completed_requests", 0)

        # [修改后] 增加重新计算逻辑
        total_task_requests = self.current_task_config.get("total_requests", 0)
        previous_completed_requests = self.current_task_config.get(
            "completed_requests", 0
        )

        # [关键修复] 如果读取到的总请求数为0（可能是旧任务文件），则根据配置重新计算
        if total_task_requests == 0:
            try:
                sample_count = self.current_task_config["config"]["sample_count"]
                eval_times = self.current_task_config["config"]["eval_times"]
                num_datasets = len(self.current_task_config["datasets"])
                num_models = len(self.current_task_config["models"])
                total_task_requests = (
                    num_models * num_datasets * sample_count * eval_times
                )
                # 更新回配置中，避免下次还为0
                self.current_task_config["total_requests"] = total_task_requests
            except Exception as e:
                pass

        # 创建评测线程，传入全局进度参数
        self.task_evaluation_thread = EvaluationThread(
            evaluator,
            datasets,
            list(selected_models.keys()),
            config,
            model_max_requests=selected_models,
            task_name=self.current_task_config.get("task_name", "unknown_task"),
            total_requests=total_task_requests,  # 确保这里传入的是非0值
            previous_completed=previous_completed_requests,
        )

        # 连接信号
        self.task_evaluation_thread.progress_updated.connect(self.update_task_progress)
        self.task_evaluation_thread.result_updated.connect(self.update_task_result)
        self.task_evaluation_thread.evaluation_finished.connect(
            self.task_evaluation_finished
        )

        # 更新UI状态
        self._stop_in_progress = False
        self._set_task_evaluation_ui_state("Running")
        self.task_progress_label.setText("准备开始...")
        self.task_result_text.clear()

        # 启动评测线程
        self.task_evaluation_thread.start()

    def stop_task_evaluation(self):
        """停止执行测评任务"""
        thread = getattr(self, "task_evaluation_thread", None)
        if thread is None or not thread.isRunning() or self._stop_in_progress:
            return

        self._stop_in_progress = True
        thread.request_stop()
        self._set_task_evaluation_ui_state("Stopping")
        self.task_progress_label.setText("正在停止...")

    def _set_task_evaluation_ui_state(self, state):
        if state not in ("Idle", "Running", "Stopping"):
            raise ValueError(f"Unsupported task evaluation state: {state}")

        self._task_evaluation_state = state
        self.start_task_button.setEnabled(state == "Idle")
        self.stop_task_button.setEnabled(state == "Running")

    def update_task_progress(self, current, total):
        """更新任务进度"""
        self.task_progress_label.setText(f"执行中... {current}/{total}")

    def update_task_result(self, result):
        """更新任务结果"""
        model_name = result.get("model", "")
        score = result.get("score", 0)
        self.task_result_text.append(f"模型 {model_name} 当前分数: {score}")

    def task_evaluation_finished(self, results):
        """任务评估完成"""
        # 更新UI状态
        self._stop_in_progress = False
        self._set_task_evaluation_ui_state("Idle")

        # 检查是否有错误
        if "error" in results:
            self.task_progress_label.setText("任务出错")
            QMessageBox.critical(
                self, "错误", f"任务执行过程中出现错误: {results['error']}"
            )
            return

        # 显示结果
        scores_file = results.get("scores_file", "")
        details_file = results.get("details_file", "")
        raw_data_file = results.get("raw_data_file", "")  # 获取详细结果文件路径
        completed_requests = results.get("completed_requests", 0)
        total_requests = self.current_task_config.get("total_requests", 0)

        # 补齐缺失的进度字段（任务文件不再存储进度）
        # Calculate actual total from completed_models when loading config
        if "completed_models" not in self.current_task_config or not isinstance(
            self.current_task_config.get("completed_models"), dict
        ):
            # Sum completed requests across all models
            actual_total = sum(
                self.current_task_config.get("completed_models", {}).values()
            )
            # Store it for progress calculation
            self.current_task_config["completed_requests"] = actual_total
        if "completed_requests" not in self.current_task_config or not isinstance(
            self.current_task_config.get("completed_requests"), (int, float)
        ):
            self.current_task_config["completed_requests"] = 0

        # 更新任务配置中的完成情况 - 累加本次完成的请求数
        previous_completed = self.current_task_config.get("completed_requests", 0)
        self.current_task_config["completed_requests"] = (
            previous_completed + completed_requests
        )

        # 更新已完成的模型信息
        for model_name, score in results.get("scores", {}).items():
            if model_name not in self.current_task_config["completed_models"]:
                self.current_task_config["completed_models"][model_name] = {}

            # 从results中获取每个模型本次实际完成的请求数（这是增量值）
            model_actual_completed = results.get("model_completed_requests", {}).get(
                model_name, 0
            )

            # 获取该模型之前已完成的请求数
            previous_model_completed = self.current_task_config["completed_models"][
                model_name
            ].get("completed_requests", 0)

            # 累加本次完成的请求数
            self.current_task_config["completed_models"][model_name][
                "completed_requests"
            ] = previous_model_completed + model_actual_completed

            # 每个模型的总请求数应该是独立的
            samples_per_model = (
                len(self.current_task_config["datasets"])
                * self.current_task_config["config"]["sample_count"]
                * self.current_task_config["config"]["eval_times"]
            )
            self.current_task_config["completed_models"][model_name][
                "total_requests"
            ] = samples_per_model

        # 检查任务是否已完成
        # 修复：任务完成的条件应该是所有模型都已完成，而不仅仅是总请求数达到
        total_completed_now = previous_completed + completed_requests
        task_completed = False

        # 计算所有模型的总请求数
        all_models_total_requests = 0
        for model_name in self.current_task_config["models"]:
            samples_per_model = (
                len(self.current_task_config["datasets"])
                * self.current_task_config["config"]["sample_count"]
                * self.current_task_config["config"]["eval_times"]
            )
            all_models_total_requests += samples_per_model

        # 检查是否所有模型都已完成
        all_models_completed = True
        for model_name in self.current_task_config["models"]:
            # 获取该模型的总请求数
            samples_per_model = (
                len(self.current_task_config["datasets"])
                * self.current_task_config["config"]["sample_count"]
                * self.current_task_config["config"]["eval_times"]
            )
            # 获取该模型已完成的请求数
            model_completed = (
                self.current_task_config["completed_models"]
                .get(model_name, {})
                .get("completed_requests", 0)
            )
            # 如果任何一个模型未完成，则任务未完成
            if model_completed < samples_per_model:
                all_models_completed = False
                break

        # 任务完成的条件：所有模型都已完成，或者总请求数达到（兼容旧逻辑）
        if all_models_completed or total_completed_now >= all_models_total_requests:
            self.current_task_config["is_completed"] = True
            # 将任务添加到已完成任务集合中
            task_filename = self.task_file_combo.currentText()
            self.completed_tasks.add(task_filename)
            # 保存配置以更新已完成任务列表 - 使用静默保存，不显示提示
            self.auto_save_config()
            task_completed = True

        # 保存更新后的任务配置（不写入完成进度到任务文件）
        task_filename = self.task_file_combo.currentText()
        task_filepath = os.path.join(self.output_dir, task_filename)
        if task_filename:
            try:
                file_task_config = self.current_task_config.copy()
                file_task_config.pop("completed_requests", None)
                file_task_config.pop("completed_models", None)
                with open(task_filepath, "w", encoding="utf-8") as f:
                    json.dump(file_task_config, f, ensure_ascii=False, indent=2)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"保存任务进度失败: {str(e)}")

        # 如果任务已完成，只删除temp文件夹，保留evaluation_task文件和任务文件夹
        if task_completed:
            try:
                import shutil

                # 检查任务文件路径是否为 "task_folder/filename" 格式
                # 使用 os.path.normpath 来标准化路径分隔符，然后检查是否包含文件夹
                normalized_path = os.path.normpath(task_filename)
                path_parts = normalized_path.split(os.sep)
                if len(path_parts) == 2:
                    task_folder, filename = path_parts
                    task_folder_path = os.path.join(self.output_dir, task_folder)

                    # 删除temp文件夹（如果存在），保留evaluation_task文件和任务文件夹
                    temp_dir = os.path.join(task_folder_path, "temp")
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

                    # 刷新任务文件列表
                    self.refresh_task_files()

            except Exception as e:
                pass

        # 更新剩余请求数显示
        self.update_remaining_requests_display()

        # 计算剩余请求数
        total_completed_now = self.current_task_config.get("completed_requests", 0)
        remaining_requests = total_requests - total_completed_now
        if remaining_requests < 0:
            remaining_requests = 0  # 避免显示负数

        # 更新进度显示
        self.task_progress_label.setText(f"任务完成 - 剩余请求数: {remaining_requests}")

        # 显示最终结果
        self.task_result_text.append("\n=== 任务完成 ===")
        self.task_result_text.append("模型分数:")
        for model_name, score in results.get("scores", {}).items():
            self.task_result_text.append(f"  {model_name}: {score}")

        self.task_result_text.append(f"\n本次完成请求数: {completed_requests}")
        self.task_result_text.append(f"总请求数: {total_requests}")
        self.task_result_text.append(f"剩余请求数: {remaining_requests}")

        # 处理文件路径显示 - 检查路径是否有效
        if scores_file and os.path.exists(scores_file):
            self.task_result_text.append(f"\n分数文件已保存到: {scores_file}")
        elif scores_file:
            self.task_result_text.append(f"\n分数文件已保存到: {scores_file}")
        else:
            self.task_result_text.append(f"\n分数文件保存失败")

        # 检查详细结果文件（raw_data_file）而不是details_file
        if raw_data_file and os.path.exists(raw_data_file):
            self.task_result_text.append(f"详细结果文件已保存到: {raw_data_file}")
        elif raw_data_file:
            self.task_result_text.append(f"详细结果文件已保存到: {raw_data_file}")
        else:
            self.task_result_text.append(f"详细结果文件保存失败")

        # 检查是否所有模型都已完成
        if completed_requests >= total_requests or remaining_requests <= 0:
            # 任务完全完成，显示正确信息
            if scores_file and os.path.exists(scores_file):
                QMessageBox.information(
                    self, "提示", f"所有任务已完成！\n分数文件已保存到: {scores_file}"
                )
            else:
                QMessageBox.information(
                    self, "提示", f"所有任务已完成！但文件保存可能存在问题。"
                )
            # 刷新任务文件列表以隐藏已完成的任务
            self.refresh_task_files()
        else:
            QMessageBox.information(
                self, "提示", f"本次任务已完成，剩余 {remaining_requests} 个请求待完成"
            )

        # 更新UI中的剩余请求数显示
        self.update_remaining_requests_display()

    def add_dataset(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择数据集文件", "", "JSONL Files (*.jsonl);;All Files (*)"
        )

        for file_path in file_paths:
            if file_path not in self.datasets:
                self.datasets.append(file_path)
                item = QListWidgetItem(file_path)
                self.dataset_list.addItem(item)

        # 添加数据集后自动保存配置
        if file_paths:
            self.populate_lists()  # 更新所有列表
            self.auto_save_config()

    def remove_dataset(self):
        selected_items = self.dataset_list.selectedItems()
        for item in selected_items:
            file_path = item.text()
            if file_path in self.datasets:
                self.datasets.remove(file_path)
            self.dataset_list.takeItem(self.dataset_list.row(item))

        # 删除数据集后自动保存配置
        if selected_items:
            self.populate_lists()  # 更新所有列表
            self.auto_save_config()

    def add_model(self):
        model_name, ok = QInputDialog.getText(self, "添加模型", "请输入模型名称:")
        if ok and model_name:
            if model_name not in self.models:
                self.models.append(model_name)
                item = QListWidgetItem(model_name)
                self.model_list.addItem(item)
                # Create default config for the model
                self.model_configs[model_name] = ModelConfig(model_name)

                # Automatically open model config dialog
                try:
                    dialog = ModelConfigDialog(self.model_configs[model_name], self)
                    result = dialog.exec_()
                    # 如果用户点击了保存,自动保存配置
                    if result == ModelConfigDialog.Accepted:
                        self.auto_save_config()
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    QMessageBox.critical(
                        self, "错误", f"打开模型配置对话框失败: {str(e)}"
                    )

                # 添加模型后自动保存配置
                self.populate_lists()  # 更新所有列表
                self.auto_save_config()

    def remove_model(self):
        selected_items = self.model_list.selectedItems()
        for item in selected_items:
            model_name = item.text()
            if model_name in self.models:
                self.models.remove(model_name)
                if model_name in self.model_configs:
                    del self.model_configs[model_name]
            self.model_list.takeItem(self.model_list.row(item))

        # 删除模型后自动保存配置
        if selected_items:
            self.populate_lists()  # 更新所有列表
            self.auto_save_config()

    def config_model(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个模型进行配置")
            return

        model_name = selected_items[0].text()
        # 确保模型配置存在，如果不存在则创建默认配置
        if model_name not in self.model_configs:
            self.model_configs[model_name] = ModelConfig(model_name)

        # Open model config dialog
        try:
            dialog = ModelConfigDialog(self.model_configs[model_name], self)
            result = dialog.exec_()
            # 如果用户点击了保存,自动保存配置
            if result == ModelConfigDialog.Accepted:
                self.auto_save_config()
        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"打开模型配置对话框失败: {str(e)}")

    def copy_model_config(self):
        """复制模型配置"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个模型进行复制")
            return

        original_model_name = selected_items[0].text()
        if original_model_name not in self.model_configs:
            QMessageBox.warning(self, "警告", "选择的模型没有配置信息")
            return

        # 获取新配置名称
        new_model_name, ok = QInputDialog.getText(
            self,
            "复制模型配置",
            "请输入新的模型配置名称:",
            text=f"{original_model_name}_copy",
        )
        if not ok or not new_model_name.strip():
            return

        # 检查新名称是否已存在
        if new_model_name in self.models:
            reply = QMessageBox.question(
                self,
                "确认",
                f"模型 '{new_model_name}' 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        # 复制模型配置
        original_config = self.model_configs[original_model_name]
        new_config = ModelConfig(new_model_name)
        # 复制所有配置参数
        new_config.api_base = original_config.api_base
        new_config.api_key = original_config.api_key
        new_config.temperature = original_config.temperature
        new_config.max_tokens = original_config.max_tokens
        new_config.top_p = original_config.top_p
        new_config.frequency_penalty = original_config.frequency_penalty
        new_config.presence_penalty = original_config.presence_penalty
        new_config.system_message = original_config.system_message
        new_config.request_model = original_config.request_model
        new_config.enable_concurrent = original_config.enable_concurrent
        new_config.max_concurrent_requests = original_config.max_concurrent_requests
        new_config.max_concurrent_total = original_config.max_concurrent_total
        new_config.timeout = original_config.timeout

        # 更新模型列表和配置
        if new_model_name not in self.models:
            self.models.append(new_model_name)
            item = QListWidgetItem(new_model_name)
            self.model_list.addItem(item)

        self.model_configs[new_model_name] = new_config
        self.populate_lists()  # 更新所有列表
        self.auto_save_config()
        QMessageBox.information(self, "提示", f"模型配置已复制为 '{new_model_name}'")

    def show_model_context_menu(self, position):
        """显示模型列表右键菜单"""
        item = self.model_list.itemAt(position)
        if not item:
            return

        menu = QMenu()
        rename_action = menu.addAction("重命名模型")
        copy_action = menu.addAction("复制模型配置")
        config_action = menu.addAction("配置模型")
        remove_action = menu.addAction("移除模型")

        action = menu.exec_(self.model_list.mapToGlobal(position))

        if action == rename_action:
            self.rename_model()
        elif action == copy_action:
            self.copy_model_config()
        elif action == config_action:
            self.config_model()
        elif action == remove_action:
            self.remove_model()

    def rename_model(self):
        """重命名模型配置"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个模型进行重命名")
            return

        old_model_name = selected_items[0].text()
        if old_model_name not in self.model_configs:
            QMessageBox.warning(self, "警告", "选择的模型没有配置信息")
            return

        # 获取新名称
        new_model_name, ok = QInputDialog.getText(
            self, "重命名模型", "请输入新的模型名称:", text=old_model_name
        )
        if not ok or not new_model_name.strip():
            return

        if new_model_name == old_model_name:
            return  # 名称未改变，直接返回

        # 检查新名称是否已存在
        if new_model_name in self.models:
            QMessageBox.warning(
                self, "警告", f"模型名称 '{new_model_name}' 已存在，请选择其他名称"
            )
            return

        # 更新模型列表和配置
        # 从模型列表中移除旧名称
        old_index = self.models.index(old_model_name)
        self.models[old_index] = new_model_name
        self.model_list.takeItem(self.model_list.row(selected_items[0]))
        item = QListWidgetItem(new_model_name)
        self.model_list.insertItem(old_index, item)

        # 更新配置字典
        old_config = self.model_configs[old_model_name]
        del self.model_configs[old_model_name]
        old_config.model_name = new_model_name  # 更新配置中的模型名称
        self.model_configs[new_model_name] = old_config

        # 如果在评测任务中使用了这个模型，也需要更新任务配置中的模型名称
        # 更新当前任务配置（如果存在）
        if hasattr(self, "current_task_config") and self.current_task_config:
            # 更新任务中的模型列表
            if old_model_name in self.current_task_config.get("models", []):
                models = self.current_task_config["models"]
                old_idx = models.index(old_model_name)
                models[old_idx] = new_model_name
                # 更新模型配置字典
                if old_model_name in self.current_task_config.get("model_configs", {}):
                    old_task_config = self.current_task_config["model_configs"][
                        old_model_name
                    ]
                    del self.current_task_config["model_configs"][old_model_name]
                    self.current_task_config["model_configs"][new_model_name] = (
                        old_task_config
                    )
                # 更新已完成模型信息
                if old_model_name in self.current_task_config.get(
                    "completed_models", {}
                ):
                    old_completed_info = self.current_task_config["completed_models"][
                        old_model_name
                    ]
                    del self.current_task_config["completed_models"][old_model_name]
                    self.current_task_config["completed_models"][new_model_name] = (
                        old_completed_info
                    )
                # 更新数据集信息
                if old_model_name in self.current_task_config.get("datasets", []):
                    datasets = self.current_task_config["datasets"]
                    if old_model_name in datasets:
                        old_idx = datasets.index(old_model_name)
                        datasets[old_idx] = new_model_name

        # 更新所有列表
        self.populate_lists()
        self.auto_save_config()
        QMessageBox.information(self, "提示", f"模型已重命名为 '{new_model_name}'")

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择结果输出目录")
        if directory:
            self.output_dir = directory
            self.output_dir_edit.setText(directory)
            # 修改输出目录后自动保存配置
            self.auto_save_config()

    def load_config(self):
        """从文件加载配置"""
        if not os.path.exists(self.config_file):
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 加载基本配置
            self.datasets = config.get("datasets", [])
            self.models = config.get("models", [])
            self.output_dir = config.get("output_dir", "")

            # 更新输出目录显示
            self.output_dir_edit.setText(self.output_dir)

            # 加载通用设置
            self.sample_count_spinbox.setValue(config.get("sample_count", 100))
            self.eval_times_spinbox.setValue(config.get("eval_times", 1))
            self.correct_score_spinbox.setValue(config.get("correct_score", 1.0))
            self.wrong_score_spinbox.setValue(config.get("wrong_score", 0.0))
            # 加载巡目比例设置（如果存在）
            if "turn_ratios" in config:
                turn_ratios = config["turn_ratios"]
                if hasattr(self, "early_ratio_spinbox"):
                    self.early_ratio_spinbox.setValue(turn_ratios.get("early", 0.33))
                if hasattr(self, "mid_ratio_spinbox"):
                    self.mid_ratio_spinbox.setValue(turn_ratios.get("mid", 0.33))
                if hasattr(self, "late_ratio_spinbox"):
                    self.late_ratio_spinbox.setValue(turn_ratios.get("late", 0.34))

            # 加载分数聚合方式
            score_agg = config.get("score_agg", "sum")
            index = self.score_agg_combo.findData(score_agg)
            if index >= 0:
                self.score_agg_combo.setCurrentIndex(index)

            # 加载请求超时设置
            if hasattr(self, "timeout_spinbox"):
                self.timeout_spinbox.setValue(config.get("timeout", 60.0))

            # 加载模型配置
            model_configs_data = config.get("model_configs", {})
            for model_name, model_config_data in model_configs_data.items():
                self.model_configs[model_name] = ModelConfig.from_dict(
                    model_config_data
                )

            # 加载已完成任务
            self.completed_tasks = set(config.get("completed_tasks", []))

        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载配置失败: {str(e)}")

    def save_config(self):
        """保存配置到文件"""
        try:
            # 确保输出目录变量与编辑框同步
            self.output_dir = self.output_dir_edit.text().strip()

            # 构建配置字典
            config = {
                "datasets": self.datasets.copy(),
                "models": self.models.copy(),
                "output_dir": self.output_dir,
                "sample_count": self.sample_count_spinbox.value(),
                "eval_times": self.eval_times_spinbox.value(),
                "correct_score": self.correct_score_spinbox.value(),
                "wrong_score": self.wrong_score_spinbox.value(),
                "score_agg": self.score_agg_combo.currentData(),
                "filter_empty_results": self.filter_empty_results_checkbox.isChecked(),
                "timeout": self.timeout_spinbox.value(),
                "model_configs": {},
                "completed_tasks": list(self.completed_tasks),  # 保存已完成任务
            }

            # 保存模型配置
            for model_name, model_config in self.model_configs.items():
                config["model_configs"][model_name] = model_config.to_dict()

            # 确保输出目录存在
            if self.output_dir and not os.path.exists(self.output_dir):
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"创建输出目录失败: {str(e)}")

            # 写入配置文件
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            QMessageBox.information(self, "提示", "配置保存成功")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存配置失败: {str(e)}")

    def auto_save_config(self):
        """自动保存配置（静默保存，不显示提示）"""
        try:
            # 确保输出目录变量与编辑框同步
            self.output_dir = self.output_dir_edit.text().strip()

            # 构建配置字典
            config = {
                "datasets": self.datasets.copy(),
                "models": self.models.copy(),
                "output_dir": self.output_dir,
                "sample_count": self.sample_count_spinbox.value(),
                "eval_times": self.eval_times_spinbox.value(),
                "correct_score": self.correct_score_spinbox.value(),
                "wrong_score": self.wrong_score_spinbox.value(),
                "score_agg": self.score_agg_combo.currentData(),
                "filter_empty_results": self.filter_empty_results_checkbox.isChecked(),
                "timeout": self.timeout_spinbox.value(),
                "model_configs": {},
                "completed_tasks": list(self.completed_tasks),
            }

            # 保存模型配置
            for model_name, model_config in self.model_configs.items():
                config["model_configs"][model_name] = model_config.to_dict()

            # 确保输出目录存在
            if self.output_dir and not os.path.exists(self.output_dir):
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                except Exception:
                    pass  # 静默失败

            # 写入配置文件
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

        except Exception:
            pass  # 静默失败，不打断用户操作

    def sync_config_to_task(self):
        """同步通用设置到评测配置"""
        if hasattr(self, "task_sample_count_spinbox"):
            self.task_sample_count_spinbox.setValue(self.sample_count_spinbox.value())
        if hasattr(self, "task_eval_times_spinbox"):
            self.task_eval_times_spinbox.setValue(self.eval_times_spinbox.value())
        if hasattr(self, "task_correct_score_spinbox"):
            self.task_correct_score_spinbox.setValue(self.correct_score_spinbox.value())
        if hasattr(self, "task_wrong_score_spinbox"):
            self.task_wrong_score_spinbox.setValue(self.wrong_score_spinbox.value())
        if hasattr(self, "task_score_agg_combo"):
            self.task_score_agg_combo.setCurrentIndex(
                self.score_agg_combo.currentIndex()
            )
        if hasattr(self, "task_early_ratio_spinbox") and hasattr(
            self, "early_ratio_spinbox"
        ):
            self.task_early_ratio_spinbox.setValue(self.early_ratio_spinbox.value())
        if hasattr(self, "task_mid_ratio_spinbox") and hasattr(
            self, "mid_ratio_spinbox"
        ):
            self.task_mid_ratio_spinbox.setValue(self.mid_ratio_spinbox.value())
        if hasattr(self, "task_late_ratio_spinbox") and hasattr(
            self, "late_ratio_spinbox"
        ):
            self.task_late_ratio_spinbox.setValue(self.late_ratio_spinbox.value())

    def init_task_config_from_general(self):
        """从通用设置初始化评测配置的默认值"""
        if hasattr(self, "task_sample_count_spinbox") and hasattr(
            self, "sample_count_spinbox"
        ):
            self.task_sample_count_spinbox.setValue(self.sample_count_spinbox.value())
        if hasattr(self, "task_eval_times_spinbox") and hasattr(
            self, "eval_times_spinbox"
        ):
            self.task_eval_times_spinbox.setValue(self.eval_times_spinbox.value())
        if hasattr(self, "task_correct_score_spinbox") and hasattr(
            self, "correct_score_spinbox"
        ):
            self.task_correct_score_spinbox.setValue(self.correct_score_spinbox.value())
        if hasattr(self, "task_wrong_score_spinbox") and hasattr(
            self, "wrong_score_spinbox"
        ):
            self.task_wrong_score_spinbox.setValue(self.wrong_score_spinbox.value())
        if hasattr(self, "task_score_agg_combo") and hasattr(self, "score_agg_combo"):
            self.task_score_agg_combo.setCurrentIndex(
                self.score_agg_combo.currentIndex()
            )
        if hasattr(self, "task_early_ratio_spinbox") and hasattr(
            self, "early_ratio_spinbox"
        ):
            self.task_early_ratio_spinbox.setValue(self.early_ratio_spinbox.value())
        if hasattr(self, "task_mid_ratio_spinbox") and hasattr(
            self, "mid_ratio_spinbox"
        ):
            self.task_mid_ratio_spinbox.setValue(self.mid_ratio_spinbox.value())
        if hasattr(self, "task_late_ratio_spinbox") and hasattr(
            self, "late_ratio_spinbox"
        ):
            self.task_late_ratio_spinbox.setValue(self.late_ratio_spinbox.value())

    def closeEvent(self, a0):
        """窗口关闭事件，自动保存配置（静默）"""
        self.auto_save_config()
        super().closeEvent(a0)

    def start_evaluation(self):
        """开始评测"""
        # 检查是否有数据集和模型
        if not self.datasets:
            QMessageBox.warning(self, "警告", "请先添加至少一个数据集")
            return

        if not self.models:
            QMessageBox.warning(self, "警告", "请先添加至少一个模型")
            return

        # 检查输出目录
        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请先设置结果输出目录")
            return

        # 获取选中的数据集
        selected_datasets = []
        for item in self.dataset_list.selectedItems():
            selected_datasets.append(item.text())

        # 如果没有选中数据集，则使用所有数据集
        if not selected_datasets:
            selected_datasets = self.datasets

        # 获取选中的模型
        selected_models = []
        for item in self.model_list.selectedItems():
            selected_models.append(item.text())

        # 如果没有选中模型，则使用所有模型
        if not selected_models:
            selected_models = self.models

        # 构建配置字典
        config = {
            "sample_count": self.sample_count_spinbox.value(),
            "eval_times": self.eval_times_spinbox.value(),
            "correct_score": self.correct_score_spinbox.value(),
            "wrong_score": self.wrong_score_spinbox.value(),
            "score_agg": self.score_agg_combo.currentData(),
            "output_dir": self.output_dir,
            "filter_empty_results": self.filter_empty_results_checkbox.isChecked(),
            "timeout": self.timeout_spinbox.value(),
            "turn_ratios": {
                "early": self.early_ratio_spinbox.value(),
                "mid": self.mid_ratio_spinbox.value(),
                "late": self.late_ratio_spinbox.value(),
            },
            "model_configs": {},
        }

        # 添加模型配置
        for model_name, model_config in self.model_configs.items():
            config["model_configs"][model_name] = model_config.to_dict()

        # 创建评测器
        from core.evaluator import ModelEvaluator

        evaluator = ModelEvaluator()

        # 创建评测线程
        self.evaluation_thread = EvaluationThread(
            evaluator, selected_datasets, selected_models, config
        )
        self.evaluation_thread.progress_updated.connect(self.update_progress)
        self.evaluation_thread.result_updated.connect(self.update_result)
        self.evaluation_thread.evaluation_finished.connect(self.evaluation_finished)

        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_label.setText("评测中...")
        self.result_text.clear()

        # 启动评测线程
        self.evaluation_thread.start()

    def stop_evaluation(self):
        """停止评测"""
        if hasattr(self, "evaluation_thread") and self.evaluation_thread.isRunning():
            # 请求停止，这会设置停止标志，让评估器知道应该停止
            if hasattr(self.evaluation_thread, "request_stop"):
                self.evaluation_thread.request_stop()
            else:
                # 如果线程没有request_stop方法，则使用terminate
                self.evaluation_thread.terminate()
            # 等待线程自然结束（等待正在处理的请求完成）
            self.evaluation_thread.wait()

        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("已停止")

    def update_progress(self, current, total):
        """更新进度"""
        self.progress_label.setText(f"评测中... {current}/{total}")

    def update_result(self, result):
        """更新结果"""
        model_name = result.get("model", "")
        score = result.get("score", 0)
        self.result_text.append(f"模型 {model_name} 当前分数: {score}")

    def evaluation_finished(self, results):
        """评测完成"""
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # 检查是否有错误
        if "error" in results:
            self.progress_label.setText("评测出错")
            QMessageBox.critical(
                self, "错误", f"评测过程中出现错误: {results['error']}"
            )
            return

        # 显示结果
        self.progress_label.setText("评测完成")
        scores_file = results.get("scores_file", "")
        details_file = results.get("details_file", "")

        # 显示最终结果
        self.result_text.append("\n=== 评测完成 ===")
        self.result_text.append("模型分数:")
        for model_name, score in results.get("scores", {}).items():
            self.result_text.append(f"  {model_name}: {score}")

        self.result_text.append(f"\n分数文件已保存到: {scores_file}")
        self.result_text.append(f"详细结果文件已保存到: {details_file}")

        QMessageBox.information(self, "提示", "评测完成")

    def create_task_tab(self, tab_widget):
        """创建任务管理标签页"""
        task_widget = QWidget()
        layout = QHBoxLayout()  # Change to horizontal layout for two columns

        # Left column - Create Evaluation Task
        left_column = QWidget()
        left_layout = QVBoxLayout()

        # Create task group
        create_group = QGroupBox("创建测评任务")
        create_layout = QVBoxLayout()

        # Dataset selection for evaluation
        dataset_eval_layout = QHBoxLayout()
        dataset_eval_layout.addWidget(QLabel("选择要评测的数据集:"))
        self.eval_dataset_list = QListWidget()
        self.eval_dataset_list.setSelectionMode(QListWidget.MultiSelection)
        dataset_eval_layout.addWidget(self.eval_dataset_list)
        create_layout.addLayout(dataset_eval_layout)

        # Model selection for evaluation
        model_eval_layout = QHBoxLayout()
        model_eval_layout.addWidget(QLabel("选择要评测的模型:"))
        self.eval_model_list = QListWidget()
        self.eval_model_list.setSelectionMode(QListWidget.MultiSelection)
        model_eval_layout.addWidget(self.eval_model_list)
        create_layout.addLayout(model_eval_layout)

        # Configuration from config tab
        config_group = QGroupBox("评测配置")
        config_layout = QVBoxLayout()

        # Sample count
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("抽取样本数量:"))
        self.task_sample_count_spinbox = QSpinBox()
        self.task_sample_count_spinbox.setRange(1, 10000)
        # 从通用设置获取默认值
        if hasattr(self, "sample_count_spinbox"):
            self.task_sample_count_spinbox.setValue(self.sample_count_spinbox.value())
        else:
            self.task_sample_count_spinbox.setValue(10)
        sample_layout.addWidget(self.task_sample_count_spinbox)
        sample_layout.addStretch()
        config_layout.addLayout(sample_layout)

        # Evaluation times
        eval_layout = QHBoxLayout()
        eval_layout.addWidget(QLabel("每个样本评测次数:"))
        self.task_eval_times_spinbox = QSpinBox()
        self.task_eval_times_spinbox.setRange(1, 100)
        # 从通用设置获取默认值
        if hasattr(self, "eval_times_spinbox"):
            self.task_eval_times_spinbox.setValue(self.eval_times_spinbox.value())
        else:
            self.task_eval_times_spinbox.setValue(1)
        eval_layout.addWidget(self.task_eval_times_spinbox)
        eval_layout.addStretch()
        config_layout.addLayout(eval_layout)

        # Scoring
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("正确得分:"))
        self.task_correct_score_spinbox = QDoubleSpinBox()
        self.task_correct_score_spinbox.setRange(0, 100)
        # 从通用设置获取默认值
        if hasattr(self, "correct_score_spinbox"):
            self.task_correct_score_spinbox.setValue(self.correct_score_spinbox.value())
        else:
            self.task_correct_score_spinbox.setValue(1.0)
        self.task_correct_score_spinbox.setSingleStep(0.1)
        score_layout.addWidget(self.task_correct_score_spinbox)

        score_layout.addWidget(QLabel("错误得分:"))
        self.task_wrong_score_spinbox = QDoubleSpinBox()
        self.task_wrong_score_spinbox.setRange(-100, 0)
        # 从通用设置获取默认值
        if hasattr(self, "wrong_score_spinbox"):
            self.task_wrong_score_spinbox.setValue(self.wrong_score_spinbox.value())
        else:
            self.task_wrong_score_spinbox.setValue(0.0)
        self.task_wrong_score_spinbox.setSingleStep(0.1)
        score_layout.addWidget(self.task_wrong_score_spinbox)
        score_layout.addStretch()
        config_layout.addLayout(score_layout)

        # Score aggregation
        agg_layout = QHBoxLayout()
        agg_layout.addWidget(QLabel("同样本分数计算方式:"))
        self.task_score_agg_combo = QComboBox()
        self.task_score_agg_combo.addItem("累加", "sum")
        self.task_score_agg_combo.addItem("平均", "avg")
        # 从通用设置获取默认值
        if hasattr(self, "score_agg_combo"):
            self.task_score_agg_combo.setCurrentIndex(
                self.score_agg_combo.currentIndex()
            )
        else:
            self.task_score_agg_combo.setCurrentIndex(0)
        agg_layout.addWidget(self.task_score_agg_combo)
        agg_layout.addStretch()
        config_layout.addLayout(agg_layout)

        # Turn ratios configuration for task
        task_turn_ratio_group = QGroupBox("巡目比例配置")
        task_turn_ratio_layout = QVBoxLayout()

        # Early turn ratio
        task_early_layout = QHBoxLayout()
        task_early_layout.addWidget(QLabel("早巡比例:"))
        self.task_early_ratio_spinbox = QDoubleSpinBox()
        self.task_early_ratio_spinbox.setRange(0, 1)
        self.task_early_ratio_spinbox.setValue(0.33)
        self.task_early_ratio_spinbox.setSingleStep(0.01)
        task_early_layout.addWidget(self.task_early_ratio_spinbox)
        task_early_layout.addStretch()
        task_turn_ratio_layout.addLayout(task_early_layout)

        # Mid turn ratio
        task_mid_layout = QHBoxLayout()
        task_mid_layout.addWidget(QLabel("中巡比例:"))
        self.task_mid_ratio_spinbox = QDoubleSpinBox()
        self.task_mid_ratio_spinbox.setRange(0, 1)
        self.task_mid_ratio_spinbox.setValue(0.33)
        self.task_mid_ratio_spinbox.setSingleStep(0.01)
        task_mid_layout.addWidget(self.task_mid_ratio_spinbox)
        task_mid_layout.addStretch()
        task_turn_ratio_layout.addLayout(task_mid_layout)

        # Late turn ratio
        task_late_layout = QHBoxLayout()
        task_late_layout.addWidget(QLabel("晚巡比例:"))
        self.task_late_ratio_spinbox = QDoubleSpinBox()
        self.task_late_ratio_spinbox.setRange(0, 1)
        self.task_late_ratio_spinbox.setValue(0.34)
        self.task_late_ratio_spinbox.setSingleStep(0.01)
        task_late_layout.addWidget(self.task_late_ratio_spinbox)
        task_late_layout.addStretch()
        task_turn_ratio_layout.addLayout(task_late_layout)

        task_turn_ratio_group.setLayout(task_turn_ratio_layout)
        config_layout.addWidget(task_turn_ratio_group)
        config_group.setLayout(config_layout)
        create_layout.addWidget(config_group)

        # Create task buttons
        create_button_layout = QHBoxLayout()
        self.select_all_datasets_button = QPushButton("全选数据集")
        self.select_all_datasets_button.clicked.connect(self.select_all_datasets)
        create_button_layout.addWidget(self.select_all_datasets_button)

        self.select_none_datasets_button = QPushButton("清空数据集")
        self.select_none_datasets_button.clicked.connect(self.select_none_datasets)
        create_button_layout.addWidget(self.select_none_datasets_button)

        self.select_all_models_button = QPushButton("全选模型")
        self.select_all_models_button.clicked.connect(self.select_all_models)
        create_button_layout.addWidget(self.select_all_models_button)

        self.select_none_models_button = QPushButton("清空模型")
        self.select_none_models_button.clicked.connect(self.select_none_models)
        create_button_layout.addWidget(self.select_none_models_button)

        create_layout.addLayout(create_button_layout)

        # Create and save task button
        self.create_task_button = QPushButton("创建并保存测评任务")
        self.create_task_button.clicked.connect(self.create_evaluation_task)
        create_layout.addWidget(self.create_task_button)

        create_group.setLayout(create_layout)
        left_layout.addWidget(create_group)
        left_column.setLayout(left_layout)

        # Right column - Execute Evaluation Task
        right_column = QWidget()
        right_layout = QVBoxLayout()

        # Execute task group
        execute_group = QGroupBox("执行测评任务")
        execute_layout = QVBoxLayout()

        # Task file selection
        task_file_layout = QHBoxLayout()
        task_file_layout.addWidget(QLabel("选择测评任务文件:"))
        self.task_file_combo = QComboBox()
        self.refresh_task_files()
        self.task_file_combo.currentTextChanged.connect(
            self.load_evaluation_task
        )  # 自动加载
        task_file_layout.addWidget(self.task_file_combo)

        self.refresh_task_button = QPushButton("刷新")
        self.refresh_task_button.clicked.connect(self.refresh_task_files)
        task_file_layout.addWidget(self.refresh_task_button)
        execute_layout.addLayout(task_file_layout)

        # Task details display
        self.task_details_text = QTextEdit()
        self.task_details_text.setReadOnly(True)
        self.task_details_text.setMaximumHeight(150)
        execute_layout.addWidget(QLabel("任务详情:"))
        execute_layout.addWidget(self.task_details_text)

        # 添加一个表格来显示每个模型的请求数设置
        from PySide6.QtWidgets import (
            QTableWidget,
            QTableWidgetItem,
            QHeaderView,
            QCheckBox,
        )

        self.model_requests_table = QTableWidget()
        self.model_requests_table.setColumnCount(4)  # 增加一列显示剩余请求数
        self.model_requests_table.setHorizontalHeaderLabels(
            ["选择", "模型", "本次请求数", "剩余请求数"]
        )
        # 设置表格自适应大小
        header = self.model_requests_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model_requests_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        execute_layout.addWidget(
            QLabel("各模型请求数设置 (请求数为0表示不使用该模型):")
        )
        execute_layout.addWidget(self.model_requests_table)

        # Execution buttons
        exec_button_layout = QHBoxLayout()
        self.load_task_button = QPushButton("加载任务")
        self.load_task_button.clicked.connect(self.load_evaluation_task)
        exec_button_layout.addWidget(self.load_task_button)

        self.start_task_button = QPushButton("开始执行")
        self.start_task_button.clicked.connect(self.start_task_evaluation)
        exec_button_layout.addWidget(self.start_task_button)

        self.stop_task_button = QPushButton("停止执行")
        self.stop_task_button.clicked.connect(self.stop_task_evaluation)
        self.stop_task_button.setEnabled(False)
        exec_button_layout.addWidget(self.stop_task_button)

        execute_layout.addLayout(exec_button_layout)

        execute_group.setLayout(execute_layout)
        right_layout.addWidget(execute_group)

        # Progress and results
        progress_group = QGroupBox("执行进度")
        progress_layout = QVBoxLayout()

        self.task_progress_label = QLabel("未开始")
        progress_layout.addWidget(self.task_progress_label)

        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)

        # Results
        result_group = QGroupBox("执行结果")
        result_layout = QVBoxLayout()

        self.task_result_text = QTextEdit()
        self.task_result_text.setReadOnly(True)
        # 设置文本框自适应大小
        self.task_result_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_layout.addWidget(self.task_result_text)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        right_column.setLayout(right_layout)

        # Add columns to main layout
        layout.addWidget(left_column)
        layout.addWidget(right_column)

        task_widget.setLayout(layout)
        tab_widget.addTab(task_widget, "测评")

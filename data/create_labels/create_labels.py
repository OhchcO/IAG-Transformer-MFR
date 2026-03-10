import os
import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QDialog, QFileDialog, QMessageBox, QSpinBox, QLabel, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox
# OCC 相关库
from OCC.Display.backend import load_backend
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Core.TopoDS import topods
from OCC.Extend.TopologyUtils import TopologyExplorer

# 你的工具库 (保留原有的加载函数)
from data_utils import load_body_from_step
from topologyCheker import TopologyChecker

load_backend("qt-pyqt5")
import OCC.Display.qtDisplay as qtDisplay

class App(QDialog):
    def __init__(self):
        super().__init__()
        # 1. 【修改点】精简初始化，增加标签存储
        self.title = "机匣特征标注工具"
        self.width, self.height = 1200, 800

        self.ais_shape = None
        self.file_name = None
        self.faces_list = []    # 存储模型所有的面对象
        self.labels = []        # 【核心】存储每个面对应的标签数字
        self.current_label = 1  # 当前选中的标签值

        self.topoChecker = TopologyChecker()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(self.width, self.height)

        # 整体采用水平布局
        windowLayout = QHBoxLayout(self)

        # 先创建左侧面板
        self.createSidePanel()

        # 再创建右侧画布
        self.canvas = qtDisplay.qtViewer3d(self)
        self.canvas.InitDriver()
        self.display = self.canvas._display
        self.display.register_select_callback(self.on_select_face)  # 7.5.1 版本写法

        # 将面板和画布按 1:5 的比例加入窗口
        windowLayout.addWidget(self.side_panel, 1)
        windowLayout.addWidget(self.canvas, 5)

        self.setLayout(windowLayout)
        self.show()

  # 记得在文件顶部导入

    def createSidePanel(self):
        self.side_panel = QWidget()
        v_layout = QVBoxLayout(self.side_panel)
        v_layout.setAlignment(Qt.AlignTop)

        # --- 第一组：文件操作 ---
        file_group = QGroupBox("文件管理")
        file_layout = QVBoxLayout()
        btn_load = QPushButton("加载 STEP 模型")
        btn_load.clicked.connect(self.openShape)
        file_layout.addWidget(btn_load)
        file_group.setLayout(file_layout)
        v_layout.addWidget(file_group)

        # --- 第二组：标注控制 ---
        label_group = QGroupBox("标注设置")
        label_layout = QGridLayout()
        label_layout.addWidget(QLabel("当前标签值:"), 0, 0)

        self.label_spin = QSpinBox()
        self.label_spin.setRange(0, 99)
        self.label_spin.setValue(1)
        self.label_spin.setMinimumHeight(30)  # 让输入框大一点好点
        self.label_spin.valueChanged.connect(self.update_label_value)
        label_layout.addWidget(self.label_spin, 0, 1)

        label_group.setLayout(label_layout)
        v_layout.addWidget(label_group)
        # --- 在标注控制组里添加重置按钮 ---
        btn_reset = QPushButton("重置所有标签")
        btn_reset.setMinimumHeight(30)
        # 使用黄色或橙色提醒用户这是一个“清除”操作
        btn_reset.setStyleSheet("background-color: #FFA000; color: white; font-weight: bold;")
        btn_reset.clicked.connect(self.reset_labels)
        label_layout.addWidget(btn_reset, 1, 0, 1, 2)  # 跨两列显示


        # --- 第三组：导出 ---
        export_group = QGroupBox("数据导出")
        export_layout = QVBoxLayout()
        btn_export = QPushButton("导出 JSON 结果")
        btn_export.setMinimumHeight(40)
        btn_export.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_export.clicked.connect(self.export_json)
        export_layout.addWidget(btn_export)
        export_group.setLayout(export_layout)
        v_layout.addWidget(export_group)
        # --- 在导出组添加校验按钮 ---
        btn_verify = QPushButton("分类验证")
        btn_verify.setMinimumHeight(35)
        # 设置一个醒目的蓝色
        btn_verify.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        btn_verify.clicked.connect(self.visual_verify)
        export_layout.addWidget(btn_verify)

        # 底部添加一个说明文本
        info_label = QLabel("\n标注说明:\n1. 选中面变红色\n2. 已标面变蓝色\n3. 标签0代表未标注")
        info_label.setStyleSheet("color: #666; font-size: 30px;")
        v_layout.addWidget(info_label)

    def update_label_value(self, value):
        """同步 QSpinBox 的值到变量"""
        self.current_label = value

    def openShape(self):
        """加载模型并初始化标签数组"""
        default_path = r"C:\Users\ROGGOR\Desktop\casing_data"
        self.file_name = QFileDialog.getOpenFileName(self, "选择模型", default_path, "*.st*p")[0]
        if not self.file_name:
            return

        solid = load_body_from_step(self.file_name)
        if not self.topoChecker(solid):
            QMessageBox.warning(self, "错误", "模型加载失败")
            return

        # 清除旧显示
        if self.ais_shape:
            self.display.Context.Erase(self.ais_shape, True)

        # 获取面列表并初始化全0标签
        topo = TopologyExplorer(solid)
        self.faces_list = list(topo.faces())
        self.labels = [0] * len(self.faces_list)

        # 显示模型
        self.ais_shape = AIS_ColoredShape(solid)
        self.display.Context.Display(self.ais_shape, True)
        self.display.SetSelectionModeFace()
        self.display.FitAll()

    def on_select_face(self, shapes, *args):
        if not self.ais_shape:
            return

        from OCC.Core.TopoDS import topods

        # --- 第一步：刷新所有“已标注”面的颜色为蓝色 ---
        # 遍历标签数组，只要值不为 0，就说明已经打过标了
        for idx, label_val in enumerate(self.labels):
            if label_val > 0:
                face_obj = self.faces_list[idx]
                # 设置为蓝色 (0, 0, 1)
                self.ais_shape.SetCustomColor(face_obj, rgb_color(0, 0, 1))

        # --- 第二步：处理“当前选中”的面，并设为红色 ---
        for shape in shapes:
            selected_face = topods.Face(shape)

            # 寻找索引
            found_idx = -1
            for idx, face in enumerate(self.faces_list):
                if face.IsEqual(selected_face):
                    found_idx = idx
                    break

            if found_idx != -1:
                # 记录标签数据
                self.labels[found_idx] = self.current_label
                print(f"面 {found_idx} 正在操作中，变红...")

                # 设置为红色 (1, 0, 0)
                self.ais_shape.SetCustomColor(selected_face, rgb_color(1, 0, 0))

        # --- 第三步：最后统一刷新渲染 ---
        self.display.Context.Redisplay(self.ais_shape, True)

    def reset_labels(self):
        """清除所有已打的标签并重置颜色"""
        if not self.ais_shape or not self.labels:
            return

        # 弹出确认框，防止手抖误点
        reply = QMessageBox.question(self, '确认', '确定要清除当前所有标注数据吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 1. 数据重置
            self.labels = [0] * len(self.faces_list)

            # 2. 视觉重置
            # ClearCustomAspects 会移除所有 SetCustomColor 设置的颜色
            self.ais_shape.ClearCustomAspects()

            # 3. 刷新视图
            self.display.Context.Redisplay(self.ais_shape, True)
            print("所有标签已重置为 0")
    def export_json(self):
        if not self.file_name or not self.labels:
            return

        pure_name = os.path.basename(self.file_name).split('.')[0]
        data = {
            "file_name": pure_name,
            "labels": self.labels
        }
        default_save_dir = os.path.join(os.path.dirname(self.file_name), "labels")

        # 如果文件夹不存在则创建
        if not os.path.exists(default_save_dir):
            os.makedirs(default_save_dir)

        save_target = os.path.join(default_save_dir, f"{pure_name}.json")
        path, _ = QFileDialog.getSaveFileName(self, "保存标注", save_target, "*.json")
        if path:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(self, "成功", "标注已导出！")

    def get_color_from_label(self, label_id):
        """根据标签 ID 自动计算一个唯一的 RGB 颜色"""
        if label_id <= 0:
            return None

        # 使用哈希种子生成伪随机数，保证颜色固定
        import random
        random.seed(label_id)
        r = random.random()
        g = random.random()
        b = random.random()

        return rgb_color(r, g, b)

    def visual_verify(self):
        """验证功能：将相同标签的面刷成相同颜色"""
        if not self.ais_shape or not self.labels:
            QMessageBox.warning(self, "提示", "请先加载模型并进行标注！")
            return

        # 1. 先清除之前所有的临时颜色（红色/蓝色）
        self.ais_shape.ClearCustomAspects()

        # 2. 遍历标签数组进行着色
        labeled_count = 0
        for idx, label_val in enumerate(self.labels):
            if label_val > 0:
                labeled_count += 1
                face_obj = self.faces_list[idx]
                # 获取该标签对应的唯一颜色
                color = self.get_color_from_label(label_val)
                self.ais_shape.SetCustomColor(face_obj, color)

        # 3. 刷新显示
        if labeled_count > 0:
            self.display.Context.Redisplay(self.ais_shape, True)
            print(f"验证模式：已对 {labeled_count} 个已标注面进行分类着色。")
        else:
            QMessageBox.information(self, "提示", "当前没有已标注的面。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
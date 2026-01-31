# 基于 YOLO11 与 RGB-D 融合的生猪关键点检测及 3D 空间定位研究

## 项目简介

本项目为技术研究报告的配套代码与实验结果，研究内容为**生猪关键点检测**与**3D 空间定位**，采用 YOLO11 姿态估计模型与 Intel RealSense RGB-D 深度相机进行融合，实现猪只实时关键点检测与三维空间坐标获取。
数据集地址为https://aistudio.baidu.com/datasetdetail/132390

## 技术框架

| 模块 | 技术选型 |
|------|----------|
| 关键点检测 | YOLO11-Pose (Ultralytics) |
| 深度感知 | Intel RealSense D400 系列 |
| RGB-D 融合 | 深度对齐 + 相机内参反投影 |
| 编程语言 | Python 3.x |

## 项目结构

```
runs/
├── 6.py                          # 实时推理主程序（RGB-D 融合 + 3D 定位）
├── README.md                     # 本说明文档
└── pig_pose/                     # 生猪姿态模型训练结果
    └── yolov11_pig_kpt/
        ├── weights/
        │   ├── best.pt           # 最优权重（推理使用）
        │   └── last.pt           # 末轮权重
        ├── args.yaml             # 训练配置
        ├── results.csv           # 训练指标
        ├── results.png           # 训练曲线
        ├── Box*_curve.png        # 检测框指标曲线
        ├── Pose*_curve.png       # 姿态关键点指标曲线
        ├── confusion_matrix*.png # 混淆矩阵
        ├── train_batch*.jpg      # 训练样本可视化
        ├── val_batch*_labels.jpg # 验证集标注
        └── val_batch*_pred.jpg   # 验证集预测
```

## 关键点定义

模型检测生猪 8 个关键点，本研究中重点关注：

| 索引 | 部位 | 用途 |
|------|------|------|
| 7 | 肩胛部 (Withers) | 躯干参考点，获取 3D 坐标 (X, Y, Z) |
| 0–5 | 后肢 (Back Legs) | 凸包区域估计，获取深度 Z |

## 环境依赖

```bash
# 核心依赖
ultralytics>=8.0
pyrealsense2
opencv-python
numpy
torch
```

## 使用方法

### 1. 硬件要求

- Intel RealSense D400 系列深度相机（或兼容型号）
- USB 3.0 连接

### 2. 运行实时检测

```bash
python 6.py
```

程序将：

1. 初始化 RealSense 相机（640×480，30fps）
2. 加载训练好的生猪姿态模型
3. 实时检测关键点并叠加 3D 坐标
4. 显示窗口，按 `q` 退出

### 3. 输出说明

- **紫色圆圈**：肩胛部区域（置信度 > 0.5）
- **黄色多边形**：后肢凸包区域（≥3 个有效关键点）
- **文字标注**：肩胛部 3D 坐标 (X, Y, Z)，后肢深度 Z（单位：米）

## 训练配置

模型基于 `yolo11n-pose.pt` 在生猪关键点数据集上微调，主要参数如下：

| 参数 | 值 |
|------|-----|
| 任务 | pose（关键点检测） |
| 迭代轮数 | 150 |
| Batch Size | 16 |
| 输入尺寸 | 640×640 |
| 优化器 | AdamW |

详细配置见 `pig_pose/yolov11_pig_kpt/args.yaml`。

## 训练结果（150 epoch）

| 指标 | 数值 |
|------|------|
| Box mAP50 | 96.70% |
| Box mAP50-95 | 64.93% |
| Pose mAP50 | 69.32% |
| Pose mAP50-95 | 66.53% |

## 3D 定位原理

1. **深度对齐**：将深度图与 RGB 图对齐至彩色相机坐标系
2. **像素 → 3D**：利用 RealSense 相机内参与 `rs2_deproject_pixel_to_point` 将像素坐标 + 深度值转换为相机坐标系下的 3D 点 (X, Y, Z)，单位：米
3. **置信度过滤**：仅对关键点置信度 > 0.5 的检测结果进行 3D 转换

## 参考文献

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- 技术报告全文请参见配套论文/报告文档

## 许可证

本项目仅供学术研究使用。


# SGGT-Net: 基于图神经网络的群目标跟踪系统

## 📋 项目简介

SGGT-Net 是一个基于图神经网络（GNN）的群目标跟踪系统，结合了扩展卡尔曼滤波（EKF），用于复杂场景下的多目标跟踪。

### 核心特性

- 🎯 **群目标跟踪**: 利用图结构建模群内目标的交互关系
- 🧠 **残差补偿**: 通过基于图的自适应频谱滤波器（GASF）提高预测精度
- 🔍 **结构学习**: 自适应学习群结构和目标间的关系
- 🔄 **卡尔曼滤波**: 集成扩展卡尔曼滤波进行状态更新

### 应用场景

- 交通场景下的车辆跟踪
- 人群场景中的行人跟踪
- 复杂环境中的多目标跟踪
- 需要考虑目标间交互的场景

---

## 🚀 快速开始

### 环境要求

```
Python >= 3.10
PyTorch >= 1.10
PyTorch Geometric >= 2.0
NumPy >= 1.19
```

### 数据准备

1. **准备数据集**
   - 支持格式：自定义轨迹数据
   - 数据应包含：目标轨迹、测量值、群标签等

2. **数据预处理**
   ```bash
   python data/preprocess_data.py --input <your_data> --output data/processed
   ```

### 训练模型

```bash
# 基础训练
python main.py --data_dir <data_path> --output_dir checkpoints/

# 自定义训练参数
python main.py \
    --data_dir your_path \
    --output_dir checkpoints/ \
    --train_batch_size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --motion_model group_residual \
    --decoder_use_GASF True\
    --decoder_use_struct True
```

---

## 📁 项目结构

```
SGGT-Net/
├── model/                          # 模型定义
│   ├── sggt_net.py              # 主模型
│   ├── kalman_utils.py          # 卡尔曼滤波工具
│   ├── motion_models.py          # 运动模型
│   ├── motion_models_base.py     # 运动模型基类
│   ├── gru_gnn_residual.py     # GNN 和残差模块
│   ├── mlp.py                  # MLP 层
│   ├── gnn_layers.py           # GNN 层
│   ├── attention_mechanism.py   # 注意力机制
│   └── ode_solvers.py          # ODE 求解器
├── train/                          # 训练相关
│   ├── trainer.py               # 原始训练器
│   ├── training_utils.py         # 训练工具模块
│   └── losses.py                # 损失函数
├── data/                           # 数据处理
│   ├── load_data.py            # 数据加载
│   └── preprocess.py            # 数据预处理
├── utils/                          # 工具函数
│   ├── common/                 # 通用工具
│   │   ├── kalman_filter.py    # 卡尔曼滤波器
│   │   ├── coordinate_transformation.py
│   │   ├── plot_track.py
│   │   └── ...
│   ├── minmax_scaler.py       # 数据归一化
│   ├── compare_models.py       # 模型对比工具
│   └── ...
├── scripts/                        # 自动化脚本
│   └── ...
├── generate_trajectory/           # 轨迹生成
├── checkpoint/                    # 模型检查点
├── log/                          # 日志文件
├── config.py                     # 配置文件
├── main.py                       # 主入口
├── README.md                     # 本文件
└── OPTIMIZATION_README.md        # 优化文档索引
```

---

## ⚙️ 配置说明

### 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|----------|
| `encoder_input_size` | 编码器输入维度 | 4 |
| `encoder_hidden_size` | 编码器隐藏层维度 | 64 |
| `decoder_hidden_size` | 解码器隐藏层维度 | 64 |
| `n_mixtures` | 高斯混合数量 | 8 |
| `batch_size` | 训练批次大小 | 32 |
| `learning_rate` | 学习率 | 1e-3 |
| `delta_T` | 时间步长 | 0.04 |

### 运动模型选项

| 模型名称 | 说明 |
|----------|------|
| `group_residual` | 完整群跟踪模型（含残差、结构和 GASF）|
| `group_residual_without_struct` | 不包含结构学习的群跟踪 |
| `group_residual_without_GASF` | 不包含机动补偿的群跟踪 |
| `group_residual_without_struct_GASF` | 基础群跟踪模型 |

---

## 📊 模型架构

### 整体架构

```
输入数据
    ↓
编码器 (GRUGNNEncoder)
    - GRU-GNN 单元
    - 时间注意力
    ↓
编码器输出
    ↓
解码器 (GRUGNNDecoder)
    - 时间注意力
    - 结构学习 (SEAN)
    - 频谱滤波 (GASF)
    ↓
预测状态 + 协方差
    ↓
扩展卡尔曼滤波 (EKF)
    - 时间更新
    - 测量更新
    ↓
最终输出
```

### 核心模块

#### 1. 编码器 (Encoder)
- **GRUGNNCell**: GRU 和 GNN 的结合单元
- **时间编码**: 处理序列数据
- **混合权重**: 输出高斯混合模型的权重

#### 2. 解码器 (Decoder)
- **时间注意力**: 关注历史信息
- **结构学习**: 自适应学习目标间关系
- **机动补偿**: 频域处理残差信号
- **过程噪声**: 动态估计过程噪声协方差

#### 3. 扩展卡尔曼滤波
- **时间更新**: 预测状态协方差
- **测量更新**: 基于观测值更新状态
- **协方差归一化**: 处理不同尺度

---

## 📈 训练和评估

### 训练流程

1. **数据准备**: 滑动窗口处理，噪声协方差计算
2. **编码**: 历史轨迹编码为特征
3. **解码**: 自回归生成预测
4. **滤波**: 使用 EKF 更新状态估计
5. **损失计算**: WTA/NLL 损失 + 卡尔曼损失
6. **反向传播**: 优化模型参数

### 评估指标

- **位置 RMSE**: 预测位置与真实位置的均方根误差
- **速度 RMSE**: 预测速度与真实速度的均方根误差
- **F1 分数**: 群结构识别的 F1 分数
- **协方差校准**: 预测协方差与真实不确定性的匹配度

---

## 🔧 代码优化

### 已完成的优化

#### 1. 卡尔曼滤波工具

**新增文件**: `model/kalman_utils.py`

**包含**:
- `MatrixBuilder`: 矩阵构建工具
- `CovarianceUtils`: 协方差操作工具
- `ExtendedKalmanFilter`: EKF 实现
- `gmm_to_single_gaussian`: 高斯混合转换

#### 2. 运动模型基类

**新增文件**: `model/motion_models_base.py`

**包含**:
- `GroupTrackMotionModelBase`: 统一基类
- 4 个子类实现
- 向后兼容接口

#### 3. 训练工具模块

**新增文件**: `train/training_utils.py`

**包含**:
- `MeasurementNoiseCalculator`: 测量噪声计算
- `CovarianceNormalizer`: 协方差归一化
- `GraphDataProcessor`: 图数据处理
- `TrainingMetrics`: 训练指标收集

### API 文档

主要模块的 API 文档：

#### SGGT_Net 模型

```python
from model.sggt_net import SGGT_Net

# 创建模型
model = SGGT_Net(
   encoder_input_size=4,
   encoder_hidden_size=64,
   encoder_n_heads=3,
   encoder_n_layers=1,
   encoder_n_mixtures=8,
   encoder_dropout=0.1,
   encoder_gnn_layer="graphconv",
   encoder_use_edge_features=True,
   decoder_motion_model=motion_model,
   decoder_max_length=10,
   decoder_hidden_size=64,
   decoder_n_heads=3,
   decoder_n_layers=1,
   decoder_alpha=0.2,
   decoder_dropout=0.1,
   decoder_residual_length=[8, 16, 32],
   decoder_z_dimension=2,
   decoder_gnn_layer="graphconv",
   decoder_use_GASF=True,
   decoder_use_struct=True,
   delta_T=0.04
)

# 训练模式前向传播
outputs = model(graph_data, tf_prob=0.5)

# 推理模式
predictions, P_t, group_num = model.predict(graph_data)

# 卡尔曼更新
x_update, P_update = model.update(state_input, P_input, measurement_input)
```

#### 卡尔曼滤波工具

```python
from model.kalman_utils import ExtendedKalmanFilter, MatrixBuilder

# 创建 EKF
ekf = ExtendedKalmanFilter(delta_T=0.04)

# 时间更新
P_next, Q_t = ekf.time_update(P_t, G_t, q_t)

# 测量更新
X_update, P_update, innovation, S = ekf.measurement_update(X_pred, P_pred, Z, R)
```

---

## 🚀 使用示例

### 基础训练

```python
from train.trainer import train_and_evaluation
from model.sggt_net import SGGT_Net
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from config import CONFIG
from data.load_data import *

# 创建运动模型
motion_model = SecondOrderNeuralODE_groupTrack(
   solver='rk4', dt=0.04, n_states=4, mixtures=8,
   n_hidden=32, n_layers=2
)

# 创建主模型
model = SGGT_Net(
   encoder_input_size=4, encoder_hidden_size=64,
   encoder_n_heads=3, encoder_n_layers=1, encoder_n_mixtures=8,
   encoder_dropout=0.1, encoder_gnn_layer="graphconv",
   encoder_use_edge_features=True,
   decoder_motion_model=motion_model,
   decoder_max_length=10, decoder_hidden_size=64,
   decoder_n_heads=3, decoder_n_layers=1, decoder_alpha=0.2,
   decoder_dropout=0.1, decoder_residual_length=[8, 16, 32],
   decoder_z_dimension=2, decoder_gnn_layer="graphconv",
   decoder_use_GASF=True, decoder_use_struct=True, delta_T=0.04
)

# 优化器和调度器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
   optimizer, factor=0.5, patience=10
)

# 加载数据
train_dataset = GroupTargetTrackingDataset_load_whole_trajectory('data/', 'train')
test_dataset = GroupTargetTrackingDataset_load_whole_trajectory('data/', 'test')

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 开始训练
train_and_evaluation(model, train_dataloader, test_dataloader,
                     optimizer, lr_schedule, logger, CONFIG, args)
```

### 模型推理

```python
import torch
from model.sggt_net import SGGT_Net
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from data.load_data import *

# 加载模型
model = SGGT_Net(...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 创建运动模型
motion_model = SecondOrderNeuralODE_groupTrack(...)
model.decoder.motion_model = motion_model

# 加载测试数据
test_dataset = GroupTargetTrackingDataset_load_whole_trajectory('data/', 'test')
test_dataloader = DataLoader(test_dataset, batch_size=1)

# 推理
with torch.no_grad():
   for graph_data in test_dataloader:
      predictions, P_t, group_num = model.predict(graph_data)
      # 处理预测结果
```

### 自定义训练流程

```python
from train.trainer import Trainer

# 创建训练器
trainer = Trainer(
   model=model,
   optimizer=optimizer,
   lr_schedule=lr_schedule,
   logger=logger,
   config=CONFIG,
   args=args
)

# 自定义训练循环
for epoch in range(100):
   # 训练一个 epoch
   train_metrics = trainer.train_epoch(train_dataloader, epoch)

   # 验证
   val_metrics = trainer.validate(val_dataloader, epoch)

   # 自定义处理
   if val_metrics['loss'] < best_loss:
      torch.save({
         'state_dict': model.state_dict(),
         'epoch': epoch,
         'metrics': val_metrics
      }, f'checkpoints/best_epoch_{epoch}.pth')
```

---

## 🗓️ 更新日志

### v1.0 (初始版本)
- 🎉 项目初始化
- 🧠 实现基础群跟踪模型
- 📊 添加训练和评估代码

---

**最后更新**: 2026-05-06

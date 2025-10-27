# 明渠边坡稳定性监测预测系统

一个完整的明渠水力学、水质和边坡稳定性集成仿真系统。

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd workspace

# 安装依赖
pip install numpy scipy matplotlib

# 可选：深度学习功能
pip install torch
```

### 运行示例

```bash
cd examples
python channel_stability_example.py
```

这将运行一个完整的20km明渠仿真案例，包括：
- 41个断面的水动力学计算
- 5种水质指标模拟（DO, BOD, NH3-N, TN, TP）
- 边坡稳定性评估
- 结果可视化

## 系统架构

```
明渠边坡稳定性系统
├── 精细化物理模型
│   ├── 水动力学（Preissmann方法 + 自适应网格）
│   ├── 水质模拟（对流-扩散-反应方程）
│   └── 边坡稳定性（极限平衡法）
│
└── 降阶模型
    ├── 线性化降阶（ID模型 + 线性水质 + 线性边坡）
    └── 深度学习降阶（LSTM/Transformer）
```

## 主要功能

### 1. 水动力学模拟

- ✅ Preissmann隐式格式求解器
- ✅ 自适应断面划分
- ✅ 多种边界条件（流量过程、水位过程、水位-流量关系）
- ✅ 稳定性好，适用于各种流态

### 2. 水质模拟

- ✅ 对流-扩散-反应方程
- ✅ 支持多种水质指标：
  - DO (溶解氧)
  - BOD (生化需氧量)
  - COD (化学需氧量)
  - NH3-N (氨氮)
  - TN (总氮)
  - TP (总磷)
  - SS (悬浮物)
- ✅ 温度修正
- ✅ 氧气限制因子

### 3. 边坡稳定性分析

- ✅ 四种失稳模式：
  - 滑动失稳
  - 倾覆失稳
  - 浮托失稳
  - 渗透失稳
- ✅ 考虑因素：
  - 地下水位
  - 渠道水位
  - 降雨强度
  - 土壤物理特性
  - 衬砌板特性
- ✅ 定量化评估（安全系数）

### 4. 监测网络集成

- ✅ 地下水观测井（每500m）
- ✅ 雨量计（渠道两端）
- ✅ 水位计（渠道两端）
- ✅ 多源数据融合

### 5. 降阶模型

#### 线性化降阶
- ✅ ID模型（瞬态演进模型）
- ✅ 线性对流扩散方程
- ✅ 线性响应边坡模型
- ✅ 计算速度快，适合实时应用

#### 深度学习降阶
- ✅ LSTM网络
- ✅ Transformer网络
- ✅ 自动特征学习
- ✅ 支持GPU加速

## 使用示例

### 基本用法

```python
from channel_stability import IntegratedSimulator, SimulationConfig
from channel_stability.core.channel_system import ChannelSystem
from channel_stability.core.monitoring_network import MonitoringNetwork
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions

# 1. 创建明渠系统
channel = ChannelSystem(
    name="测试明渠",
    total_length=20000.0,  # 20 km
    sections=[],
)

# 创建均匀断面
channel.create_uniform_sections(
    num_sections=41,
    base_section=base_section,
    bed_slope=0.0001,
)

# 2. 创建监测网络
network = MonitoringNetwork(network_name="监测网络")
network.create_uniform_groundwater_network(
    channel_length=20000.0,
    spacing=500.0,
)
network.add_boundary_stations(0.0, 20000.0)

# 3. 定义边界条件
bc = BoundaryConditions.create_constant_bc(
    upstream_q=10.0,
    downstream_h=2.5,
)

# 4. 运行仿真
simulator = IntegratedSimulator(
    channel=channel,
    monitoring_network=network,
    boundary_conditions=bc,
)

config = SimulationConfig(
    total_time=86400.0,  # 24小时
    dt=60.0,  # 1分钟
    enable_hydrodynamics=True,
    enable_water_quality=True,
    enable_slope_stability=True,
)

results = simulator.run_simulation(config)

# 5. 导出结果
simulator.export_results(results, "output")
```

### 降阶模型使用

```python
from channel_stability.reduced_order import LinearizedHydrodynamics

# 线性化水动力学模型
linear_model = LinearizedHydrodynamics(
    stations=channel.stations,
    bed_elevations=bed_elevations,
    bottom_widths=bottom_widths,
)

# 求解
result = linear_model.solve_kinematic_wave(
    total_time=86400.0,
    dt=60.0,
    upstream_discharge_series=upstream_q,
)
```

## 输出结果

系统生成以下输出文件：

```
simulation_results/
├── summary.json                # 结果摘要
├── hydrodynamics.npz           # 水动力学结果
├── water_quality.npz           # 水质结果
├── slope_stability.npz         # 边坡稳定性结果
└── visualization.png           # 可视化图表
```

### 结果内容

1. **summary.json**: 
   - 最大水深、流速、Froude数
   - 各水质指标的范围
   - 最小稳定系数、不稳定断面数

2. **hydrodynamics.npz**:
   - times: 时间序列
   - stations: 断面桩号
   - depths: 水深场
   - discharges: 流量场
   - velocities: 流速场
   - water_levels: 水位场
   - froude_numbers: Froude数场

3. **water_quality.npz**:
   - times: 时间序列
   - stations: 断面桩号
   - DO, BOD, NH3N, TN, TP: 各水质指标浓度场

4. **slope_stability.npz**:
   - times: 时间序列
   - stations: 断面桩号
   - sliding_factors: 抗滑稳定系数
   - overturning_factors: 抗倾覆稳定系数
   - uplift_factors: 抗浮稳定系数
   - seepage_factors: 渗透稳定系数
   - comprehensive_factors: 综合稳定系数
   - stability_status: 稳定性状态
   - channel_stability_index: 全渠道稳定性指标

## 技术特点

1. **高精度物理模型**
   - Preissmann隐式格式（无条件稳定）
   - 自适应网格技术
   - 多物理场耦合

2. **实用性强**
   - 考虑实际监测数据
   - 支持多种边界条件
   - 断面独立建模，计算高效

3. **定量化评估**
   - 多层次稳定性指标
   - 时空演化分析
   - 预警阈值判定

4. **模块化设计**
   - 松耦合架构
   - 易于扩展
   - 支持部分模块单独使用

## 系统要求

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.3

可选：
- PyTorch >= 1.9（深度学习模型）
- CUDA（GPU加速）

## 性能

典型案例（20km渠道，41个断面，24小时仿真）：

| 模型类型 | 计算时间 | 精度 |
|---------|---------|------|
| 精细化模型 | ~10分钟 | 高 |
| 线性化降阶 | ~10秒 | 中 |
| 深度学习降阶 | ~1秒 | 中-高 |

*测试环境: Intel i7, 16GB RAM*

## 文档

详细技术文档请参阅：
- [技术方案](./明渠边坡稳定性监测预测系统技术方案.md)
- [API文档](./docs/api.md)（待补充）

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

[待定]

## 致谢

感谢所有贡献者和支持者！

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。

---

**版本**: 1.0.0  
**最后更新**: 2025-10-27

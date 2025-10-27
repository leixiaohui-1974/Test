# 项目开发规范

## 文档版本
- **版本**: 1.0
- **日期**: 2025-10-27
- **适用范围**: 明渠边坡稳定性监测预测系统

---

## 1. 总体架构原则

### 1.1 两层架构设计

本项目采用**两层架构**设计，确保代码的可维护性、可扩展性和可重用性。

```
项目结构
├── 基础模块库层 (Core Library Layer)
│   ├── channel_stability/         # 核心功能模块
│   ├── water_resource_system/     # 水资源系统模块
│   └── hydraulic_model/           # 水力模型模块
│
└── 应用层 (Application Layer)
    ├── examples/                  # 应用示例
    ├── tests/                     # 测试程序
    └── scripts/                   # 应用脚本
```

### 1.2 分层职责

#### **基础模块库层**
- **定位**: 提供通用的、可复用的核心功能
- **特点**: 
  - 高度抽象，参数化设计
  - 无硬编码值
  - 无特定应用场景假设
  - 完整的文档和类型注解
  - 纯粹的计算和数据处理逻辑

#### **应用层**
- **定位**: 基于基础库构建具体应用
- **特点**:
  - 可包含具体场景的配置参数
  - 可包含硬编码的示例数据
  - 组合基础库模块实现业务逻辑
  - 生成可视化结果、报告等

---

## 2. 基础模块库开发规范

### 2.1 代码组织

#### 2.1.1 模块结构

每个核心模块应遵循以下结构：

```
module_name/
├── __init__.py              # 模块导出接口
├── core/                    # 核心功能
│   ├── __init__.py
│   ├── data_structures.py   # 数据结构定义
│   └── algorithms.py        # 算法实现
├── utils/                   # 工具函数
│   ├── __init__.py
│   └── helpers.py
└── README.md               # 模块文档
```

#### 2.1.2 命名规范

- **模块名**: 使用小写字母和下划线 (snake_case)
  - 示例: `slope_stability`, `water_quality`

- **类名**: 使用大驼峰命名 (PascalCase)
  - 示例: `ChannelSystem`, `SlopeStabilityCalculator`

- **函数名**: 使用小写字母和下划线 (snake_case)
  - 示例: `calculate_safety_factor`, `solve_hydrodynamics`

- **常量名**: 使用大写字母和下划线 (UPPER_SNAKE_CASE)
  - 示例: `DEFAULT_GRAVITY`, `MAX_ITERATIONS`

### 2.2 参数化设计原则

#### ❌ 错误示例（硬编码）

```python
def calculate_flow(depth):
    """计算流量 - 错误示例"""
    width = 6.0  # ❌ 硬编码渠道宽度
    manning_n = 0.025  # ❌ 硬编码糙率
    slope = 0.0002  # ❌ 硬编码坡度
    
    area = width * depth
    velocity = (1/manning_n) * (area / (width + 2*depth))**(2/3) * slope**0.5
    return area * velocity
```

#### ✅ 正确示例（参数化）

```python
def calculate_flow(
    depth: float,
    width: float,
    manning_n: float,
    bed_slope: float,
    gravity: float = 9.81
) -> float:
    """
    计算明渠流量
    
    Parameters
    ----------
    depth : float
        水深 (m)
    width : float
        渠底宽度 (m)
    manning_n : float
        曼宁糙率系数
    bed_slope : float
        渠底坡度
    gravity : float, optional
        重力加速度 (m/s²), 默认 9.81
        
    Returns
    -------
    float
        流量 (m³/s)
    """
    area = width * depth
    wetted_perimeter = width + 2 * depth
    hydraulic_radius = area / wetted_perimeter
    velocity = (1/manning_n) * hydraulic_radius**(2/3) * bed_slope**0.5
    return area * velocity
```

### 2.3 配置与数据分离

#### 2.3.1 使用配置类

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SolverConfig:
    """求解器配置"""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    relaxation_factor: float = 0.5
    verbose: bool = False
    
@dataclass
class ChannelGeometry:
    """渠道几何参数"""
    length: float
    width: float
    depth: float
    slope: float
    manning_n: float
```

#### 2.3.2 配置文件支持

基础库应支持从外部配置文件加载参数：

```python
import json
from pathlib import Path
from typing import Dict, Any

def load_config(config_file: Path) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Parameters
    ----------
    config_file : Path
        配置文件路径
        
    Returns
    -------
    Dict[str, Any]
        配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)
```

### 2.4 默认值规范

#### 2.4.1 物理常数

物理常数应定义在专门的常量模块中：

```python
# constants.py
"""物理和数值常量"""

# 物理常数
GRAVITY = 9.81  # 重力加速度 (m/s²)
WATER_DENSITY = 1000.0  # 水密度 (kg/m³)
WATER_VISCOSITY = 1e-6  # 水动力粘度 (m²/s)

# 数值常数
EPSILON = 1e-10  # 数值零值判断
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_TOLERANCE = 1e-6
```

#### 2.4.2 默认参数使用原则

- **必须参数**: 没有合理默认值的参数（如几何尺寸、边界条件）
- **可选参数**: 有通用默认值的参数（如重力加速度、收敛容差）

```python
def solve_equation(
    # 必须参数 - 无默认值
    initial_condition: np.ndarray,
    boundary_conditions: BoundaryConditions,
    
    # 可选参数 - 有默认值
    dt: float = 1.0,
    max_time: float = 3600.0,
    tolerance: float = 1e-6,
    gravity: float = GRAVITY,
):
    """求解方程"""
    pass
```

### 2.5 文档规范

#### 2.5.1 模块级文档

每个模块文件应包含：

```python
"""
模块名称

简短描述（一行）

详细描述（多行）
- 功能说明
- 使用场景
- 依赖关系

作者: XXX
日期: YYYY-MM-DD
"""
```

#### 2.5.2 函数文档（NumPy风格）

```python
def function_name(
    param1: Type1,
    param2: Type2,
    optional_param: Type3 = default_value
) -> ReturnType:
    """
    函数功能简述（一行）
    
    详细功能描述（可选，多行）
    
    Parameters
    ----------
    param1 : Type1
        参数1的描述
    param2 : Type2
        参数2的描述
    optional_param : Type3, optional
        可选参数的描述，默认值 default_value
        
    Returns
    -------
    ReturnType
        返回值描述
        
    Raises
    ------
    ValueError
        描述何时抛出此异常
    RuntimeError
        描述何时抛出此异常
        
    Examples
    --------
    >>> result = function_name(value1, value2)
    >>> print(result)
    expected_output
    
    Notes
    -----
    - 额外说明1
    - 额外说明2
    
    References
    ----------
    [1] 相关文献或资料
    """
    # 实现代码
    pass
```

#### 2.5.3 类文档

```python
class ClassName:
    """
    类功能简述
    
    详细功能描述
    
    Attributes
    ----------
    attr1 : Type1
        属性1的描述
    attr2 : Type2
        属性2的描述
        
    Methods
    -------
    method1(param)
        方法1的简短描述
    method2(param)
        方法2的简短描述
        
    Examples
    --------
    >>> obj = ClassName(param1, param2)
    >>> result = obj.method1()
    """
    
    def __init__(self, param1: Type1, param2: Type2):
        """
        初始化类实例
        
        Parameters
        ----------
        param1 : Type1
            参数1描述
        param2 : Type2
            参数2描述
        """
        self.attr1 = param1
        self.attr2 = param2
```

### 2.6 类型注解规范

所有公共接口必须包含完整的类型注解：

```python
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

def process_data(
    input_data: np.ndarray,
    config: Dict[str, Any],
    output_file: Optional[Path] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """处理数据"""
    # 实现
    results = np.zeros_like(input_data)
    metrics = {"mean": 0.0, "std": 0.0}
    return results, metrics
```

### 2.7 错误处理

#### 2.7.1 输入验证

```python
def calculate_safety_factor(
    cohesion: float,
    friction_angle: float,
    slope_angle: float
) -> float:
    """计算安全系数"""
    
    # 输入验证
    if cohesion < 0:
        raise ValueError(f"粘聚力必须非负，当前值: {cohesion}")
    
    if not (0 <= friction_angle <= 90):
        raise ValueError(f"内摩擦角必须在0-90度之间，当前值: {friction_angle}")
    
    if not (0 <= slope_angle <= 90):
        raise ValueError(f"边坡角必须在0-90度之间，当前值: {slope_angle}")
    
    # 计算逻辑
    # ...
    
    return safety_factor
```

#### 2.7.2 异常类型

使用适当的异常类型：

- `ValueError`: 参数值不合法
- `TypeError`: 参数类型错误
- `RuntimeError`: 运行时错误（如求解失败）
- `FileNotFoundError`: 文件不存在
- 自定义异常: 特定领域的错误

```python
class ConvergenceError(RuntimeError):
    """数值求解未收敛异常"""
    pass

class InvalidGeometryError(ValueError):
    """无效的几何参数异常"""
    pass
```

### 2.8 单元测试要求

每个基础模块必须包含单元测试：

```python
# tests/unit/test_flow_calculator.py
import pytest
import numpy as np
from channel_stability.hydrodynamics.flow_calculator import calculate_flow

class TestFlowCalculator:
    """流量计算器单元测试"""
    
    def test_basic_flow_calculation(self):
        """测试基本流量计算"""
        result = calculate_flow(
            depth=2.0,
            width=5.0,
            manning_n=0.025,
            bed_slope=0.001
        )
        assert result > 0
        assert isinstance(result, float)
    
    def test_invalid_depth(self):
        """测试无效水深"""
        with pytest.raises(ValueError):
            calculate_flow(
                depth=-1.0,  # 负值水深
                width=5.0,
                manning_n=0.025,
                bed_slope=0.001
            )
    
    def test_zero_slope(self):
        """测试零坡度"""
        result = calculate_flow(
            depth=2.0,
            width=5.0,
            manning_n=0.025,
            bed_slope=0.0
        )
        assert result == 0.0
    
    @pytest.mark.parametrize("depth,width,expected", [
        (1.0, 5.0, 2.5),
        (2.0, 5.0, 7.8),
        (3.0, 5.0, 15.6),
    ])
    def test_flow_with_different_depths(self, depth, width, expected):
        """测试不同水深的流量计算"""
        result = calculate_flow(
            depth=depth,
            width=width,
            manning_n=0.025,
            bed_slope=0.001
        )
        assert abs(result - expected) < 0.5  # 允许误差
```

---

## 3. 应用层开发规范

### 3.1 应用层职责

应用层负责：
1. 组合基础库模块构建完整应用
2. 提供具体场景的配置参数
3. 生成可视化结果和报告
4. 实现用户交互界面

### 3.2 目录结构

```
examples/                          # 应用示例
├── basic_channel_flow.py         # 基础流动示例
├── slope_stability_analysis.py   # 边坡稳定性分析示例
└── configs/                      # 示例配置文件
    ├── channel_config.json
    └── solver_config.json

tests/                            # 测试程序
├── test_integration.py           # 集成测试
├── test_scenarios.py             # 场景测试
└── test_performance.py           # 性能测试

scripts/                          # 工具脚本
├── run_batch_analysis.py         # 批量分析
├── generate_report.py            # 报告生成
└── data_preprocessing.py         # 数据预处理
```

### 3.3 应用代码规范

#### 3.3.1 配置集中管理

```python
# examples/configs/channel_config.json
{
  "channel": {
    "length": 5000.0,
    "width": 6.0,
    "slope": 0.0002,
    "manning_n": 0.025
  },
  "soil": {
    "cohesion": 15.0,
    "friction_angle": 30.0,
    "unit_weight": 18.0
  },
  "simulation": {
    "total_time": 7200.0,
    "time_step": 60.0,
    "output_interval": 300.0
  }
}
```

```python
# examples/slope_stability_analysis.py
import json
from pathlib import Path
from channel_stability.core.channel_system import ChannelSystem
from channel_stability.integrated_simulation import IntegratedSimulator

def load_config(config_file: Path) -> dict:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """主程序"""
    # 加载配置
    config_file = Path(__file__).parent / "configs" / "channel_config.json"
    config = load_config(config_file)
    
    # 使用基础库创建系统
    channel = ChannelSystem(
        name="分析渠道",
        total_length=config['channel']['length'],
        sections=[]
    )
    
    # 运行仿真
    simulator = IntegratedSimulator(channel=channel, ...)
    results = simulator.run_simulation(...)
    
    # 生成报告
    generate_report(results, output_dir="./output")

if __name__ == "__main__":
    main()
```

#### 3.3.2 可视化规范

应用层负责所有可视化，基础库不应包含绘图代码：

```python
# examples/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_safety_factor_evolution(times, safety_factors, output_file):
    """
    绘制安全系数演化图
    
    Parameters
    ----------
    times : np.ndarray
        时间序列
    safety_factors : np.ndarray
        安全系数序列
    output_file : Path
        输出文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times / 3600, safety_factors, 'b-', linewidth=2)
    ax.axhline(y=1.3, color='orange', linestyle='--', label='安全阈值')
    ax.axhline(y=1.0, color='red', linestyle='--', label='临界状态')
    
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('安全系数', fontsize=12)
    ax.set_title('边坡安全系数时间演化', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
```

### 3.4 测试规范

#### 3.4.1 集成测试

```python
# tests/test_integration.py
import pytest
from pathlib import Path
from channel_stability.core.channel_system import ChannelSystem
from channel_stability.integrated_simulation import IntegratedSimulator

class TestIntegratedSystem:
    """集成测试"""
    
    def test_complete_simulation_workflow(self):
        """测试完整仿真流程"""
        # 创建系统
        channel = ChannelSystem(...)
        simulator = IntegratedSimulator(...)
        
        # 运行仿真
        results = simulator.run_simulation(...)
        
        # 验证结果
        assert results is not None
        assert results.hydrodynamics is not None
        assert results.slope_stability is not None
        
    def test_mass_conservation(self):
        """测试质量守恒"""
        # ...
```

#### 3.4.2 场景测试

```python
# tests/test_scenarios.py
import pytest

class TestFloodScenarios:
    """洪水场景测试"""
    
    @pytest.mark.parametrize("flow_increase", [1.5, 2.0, 2.5])
    def test_sudden_flow_increase(self, flow_increase):
        """测试流量突增场景"""
        # 设置场景
        # 运行仿真
        # 验证结果
        pass
```

---

## 4. 开发流程规范

### 4.1 Bug修复流程

当发现错误时，遵循以下流程：

```
1. 错误定位
   ↓
2. 判断错误位置
   ├─→ 基础库错误？
   │   ├─→ 修复基础库代码
   │   ├─→ 更新基础库单元测试
   │   └─→ 更新版本号
   │
   └─→ 应用层错误？
       ├─→ 修复应用层代码
       └─→ 更新应用层测试
       
3. 验证修复
   ├─→ 运行单元测试
   ├─→ 运行集成测试
   └─→ 运行场景测试
   
4. 文档更新
   └─→ 更新CHANGELOG
```

#### 4.1.1 基础库Bug修复示例

**场景**: 在测试中发现边坡稳定性计算结果异常

```python
# ❌ 修复前 - channel_stability/slope_stability/stability_calculator.py
def calculate_safety_factor(self, slope_angle, water_depth):
    """计算安全系数"""
    # Bug: 硬编码了土壤参数
    cohesion = 15.0  # ❌ 错误：硬编码
    friction_angle = 30.0  # ❌ 错误：硬编码
    
    fs = cohesion / (water_depth * math.tan(math.radians(slope_angle)))
    return fs
```

```python
# ✅ 修复后 - channel_stability/slope_stability/stability_calculator.py
def calculate_safety_factor(
    self,
    slope_angle: float,
    water_depth: float,
    cohesion: float,
    friction_angle: float,
    unit_weight: float
) -> float:
    """
    计算边坡安全系数
    
    Parameters
    ----------
    slope_angle : float
        边坡角度 (度)
    water_depth : float
        水深 (m)
    cohesion : float
        土壤粘聚力 (kPa)
    friction_angle : float
        内摩擦角 (度)
    unit_weight : float
        土壤重度 (kN/m³)
        
    Returns
    -------
    float
        安全系数
    """
    # 参数验证
    if cohesion < 0:
        raise ValueError(f"粘聚力必须非负: {cohesion}")
    
    # 计算逻辑
    slope_rad = math.radians(slope_angle)
    friction_rad = math.radians(friction_angle)
    
    driving_force = unit_weight * water_depth * math.sin(slope_rad)
    resisting_force = cohesion + unit_weight * water_depth * math.cos(slope_rad) * math.tan(friction_rad)
    
    fs = resisting_force / (driving_force + 1e-10)  # 避免除零
    return fs
```

```python
# ✅ 更新应用层代码 - examples/slope_stability_analysis.py
# 从配置读取参数
soil_params = config['soil']
safety_factor = calculator.calculate_safety_factor(
    slope_angle=26.57,
    water_depth=depth,
    cohesion=soil_params['cohesion'],
    friction_angle=soil_params['friction_angle'],
    unit_weight=soil_params['unit_weight']
)
```

### 4.2 新功能开发流程

```
1. 需求分析
   ├─→ 确定功能归属（基础库 or 应用层）
   └─→ 编写功能规格说明
   
2. 接口设计
   ├─→ 设计函数/类接口
   ├─→ 确定参数和返回值
   └─→ 编写接口文档
   
3. 实现
   ├─→ 编写基础库代码（如适用）
   ├─→ 编写单元测试
   ├─→ 编写应用层代码
   └─→ 编写集成测试
   
4. 验证
   ├─→ 运行所有测试
   ├─→ 代码审查
   └─→ 性能测试（如需要）
   
5. 文档化
   ├─→ 更新API文档
   ├─→ 编写使用示例
   └─→ 更新CHANGELOG
```

### 4.3 代码审查清单

#### 基础库代码审查

- [ ] 是否有硬编码的参数值？
- [ ] 所有参数是否都可配置？
- [ ] 是否有完整的类型注解？
- [ ] 是否有完整的文档字符串？
- [ ] 是否有输入验证？
- [ ] 是否有单元测试？
- [ ] 测试覆盖率是否 > 80%？
- [ ] 是否有不必要的依赖？

#### 应用层代码审查

- [ ] 是否正确使用基础库接口？
- [ ] 配置参数是否集中管理？
- [ ] 是否有清晰的输出和日志？
- [ ] 错误处理是否完善？
- [ ] 是否有集成测试？

---

## 5. 版本管理

### 5.1 版本号规范

使用语义化版本号 (Semantic Versioning): `MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的API修改
- **MINOR**: 向后兼容的功能新增
- **PATCH**: 向后兼容的问题修正

示例:
- `1.0.0` → `1.0.1`: 修复bug
- `1.0.0` → `1.1.0`: 新增功能
- `1.0.0` → `2.0.0`: 破坏性变更

### 5.2 CHANGELOG维护

每次修改都应更新CHANGELOG.md：

```markdown
# 更新日志

## [1.1.0] - 2025-10-27

### 新增
- 新增边坡稳定性动态分析功能
- 支持多种土壤本构模型

### 修改
- 优化水动力学求解器性能
- 改进质量守恒算法

### 修复
- 修复安全系数计算中的硬编码问题 (#42)
- 修复边界条件处理错误 (#38)

### 废弃
- 计划在v2.0移除旧的Preissmann求解器接口

## [1.0.1] - 2025-10-20

### 修复
- 修复水质模拟浓度计算错误
```

---

## 6. 最佳实践示例

### 6.1 完整的基础库模块示例

```python
# channel_stability/slope_stability/simple_calculator.py
"""
边坡稳定性简化计算器

提供基于无限边坡理论的快速安全系数计算。

作者: 开发团队
日期: 2025-10-27
"""

import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class SoilParameters:
    """
    土壤参数
    
    Attributes
    ----------
    cohesion : float
        粘聚力 (kPa)
    friction_angle : float
        内摩擦角 (度)
    unit_weight : float
        土壤重度 (kN/m³)
    saturated_unit_weight : float
        饱和重度 (kN/m³)
    """
    cohesion: float
    friction_angle: float
    unit_weight: float
    saturated_unit_weight: float
    
    def __post_init__(self):
        """参数验证"""
        if self.cohesion < 0:
            raise ValueError(f"粘聚力必须非负: {self.cohesion}")
        if not (0 <= self.friction_angle <= 90):
            raise ValueError(f"内摩擦角必须在0-90度: {self.friction_angle}")
        if self.unit_weight <= 0:
            raise ValueError(f"重度必须为正: {self.unit_weight}")


class SimpleSlopeCalculator:
    """
    简化边坡稳定性计算器
    
    基于无限边坡理论计算安全系数。
    
    Methods
    -------
    calculate_safety_factor(slope_angle, water_depth, soil)
        计算安全系数
    is_stable(safety_factor, threshold)
        判断边坡是否稳定
        
    Examples
    --------
    >>> soil = SoilParameters(
    ...     cohesion=15.0,
    ...     friction_angle=30.0,
    ...     unit_weight=18.0,
    ...     saturated_unit_weight=20.0
    ... )
    >>> calculator = SimpleSlopeCalculator()
    >>> fs = calculator.calculate_safety_factor(
    ...     slope_angle=26.57,
    ...     water_depth=2.0,
    ...     soil=soil
    ... )
    >>> print(f"安全系数: {fs:.2f}")
    安全系数: 1.85
    """
    
    def __init__(self, gravity: float = 9.81):
        """
        初始化计算器
        
        Parameters
        ----------
        gravity : float, optional
            重力加速度 (m/s²), 默认 9.81
        """
        self.gravity = gravity
    
    def calculate_safety_factor(
        self,
        slope_angle: float,
        water_depth: float,
        soil: SoilParameters,
        consider_seepage: bool = True
    ) -> float:
        """
        计算边坡安全系数
        
        Parameters
        ----------
        slope_angle : float
            边坡角度 (度)
        water_depth : float
            水深 (m)
        soil : SoilParameters
            土壤参数
        consider_seepage : bool, optional
            是否考虑渗流影响, 默认 True
            
        Returns
        -------
        float
            安全系数
            
        Raises
        ------
        ValueError
            如果输入参数无效
            
        Notes
        -----
        使用无限边坡理论:
        FS = (c' + (γ - γw) * h * cos²β * tanφ') / (γ * h * sinβ * cosβ)
        
        其中:
        - c': 有效粘聚力
        - φ': 有效内摩擦角
        - γ: 土壤重度
        - γw: 水的重度
        - h: 边坡高度
        - β: 边坡角度
        """
        # 输入验证
        if not (0 <= slope_angle <= 90):
            raise ValueError(f"边坡角度必须在0-90度: {slope_angle}")
        if water_depth < 0:
            raise ValueError(f"水深必须非负: {water_depth}")
        
        # 特殊情况处理
        if water_depth == 0:
            return float('inf')  # 无水深时视为完全稳定
        if slope_angle == 0:
            return float('inf')  # 水平坡面完全稳定
        
        # 单位转换
        slope_rad = math.radians(slope_angle)
        friction_rad = math.radians(soil.friction_angle)
        
        # 计算有效应力
        if consider_seepage:
            gamma_eff = soil.saturated_unit_weight - self.gravity  # 考虑浮力
        else:
            gamma_eff = soil.unit_weight
        
        # 计算驱动力和抗滑力
        sin_beta = math.sin(slope_rad)
        cos_beta = math.cos(slope_rad)
        tan_phi = math.tan(friction_rad)
        
        # 抗滑力
        cohesive_resistance = soil.cohesion
        frictional_resistance = gamma_eff * water_depth * cos_beta**2 * tan_phi
        total_resistance = cohesive_resistance + frictional_resistance
        
        # 驱动力
        driving_force = soil.unit_weight * water_depth * sin_beta * cos_beta
        
        # 安全系数
        safety_factor = total_resistance / (driving_force + 1e-10)  # 避免除零
        
        return safety_factor
    
    @staticmethod
    def is_stable(
        safety_factor: float,
        threshold: float = 1.3
    ) -> bool:
        """
        判断边坡是否稳定
        
        Parameters
        ----------
        safety_factor : float
            安全系数
        threshold : float, optional
            安全阈值, 默认 1.3
            
        Returns
        -------
        bool
            True表示稳定，False表示不稳定
            
        Notes
        -----
        典型安全阈值:
        - FS > 1.5: 非常安全
        - FS > 1.3: 安全
        - FS > 1.0: 基本稳定
        - FS < 1.0: 不稳定
        """
        return safety_factor >= threshold
```

### 6.2 完整的应用层示例

```python
# examples/simple_slope_analysis.py
"""
简单边坡稳定性分析示例

演示如何使用基础库进行边坡稳定性评估。
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from channel_stability.slope_stability.simple_calculator import (
    SimpleSlopeCalculator,
    SoilParameters
)


def load_config(config_file: Path) -> dict:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_slope_stability(config: dict, output_dir: Path):
    """
    分析边坡稳定性
    
    Parameters
    ----------
    config : dict
        配置参数
    output_dir : Path
        输出目录
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从配置加载参数
    soil_cfg = config['soil']
    soil = SoilParameters(
        cohesion=soil_cfg['cohesion'],
        friction_angle=soil_cfg['friction_angle'],
        unit_weight=soil_cfg['unit_weight'],
        saturated_unit_weight=soil_cfg['saturated_unit_weight']
    )
    
    # 创建计算器
    calculator = SimpleSlopeCalculator()
    
    # 分析不同水深的安全系数
    water_depths = np.linspace(0.5, 5.0, 50)
    slope_angle = config['slope']['angle']
    
    safety_factors = []
    for depth in water_depths:
        fs = calculator.calculate_safety_factor(
            slope_angle=slope_angle,
            water_depth=depth,
            soil=soil
        )
        safety_factors.append(fs)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(water_depths, safety_factors, 'b-', linewidth=2, label='安全系数')
    ax.axhline(y=1.5, color='green', linestyle='--', label='非常安全 (FS=1.5)')
    ax.axhline(y=1.3, color='orange', linestyle='--', label='安全阈值 (FS=1.3)')
    ax.axhline(y=1.0, color='red', linestyle='--', label='临界状态 (FS=1.0)')
    
    ax.set_xlabel('水深 (m)', fontsize=12)
    ax.set_ylabel('安全系数', fontsize=12)
    ax.set_title(f'边坡稳定性分析 (坡角={slope_angle}°)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图表
    output_file = output_dir / 'slope_stability_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    
    # 生成报告
    report_lines = []
    report_lines.append("# 边坡稳定性分析报告\n")
    report_lines.append(f"## 输入参数\n")
    report_lines.append(f"- 边坡角度: {slope_angle}°")
    report_lines.append(f"- 土壤粘聚力: {soil.cohesion} kPa")
    report_lines.append(f"- 内摩擦角: {soil.friction_angle}°")
    report_lines.append(f"- 土壤重度: {soil.unit_weight} kN/m³\n")
    
    report_lines.append(f"## 分析结果\n")
    report_lines.append(f"- 最小安全系数: {min(safety_factors):.3f}")
    report_lines.append(f"- 最大安全系数: {max(safety_factors):.3f}")
    
    # 判断稳定性
    critical_depth = None
    for depth, fs in zip(water_depths, safety_factors):
        if fs < 1.3:
            critical_depth = depth
            break
    
    if critical_depth:
        report_lines.append(f"- ⚠ 警告: 当水深超过{critical_depth:.2f}m时，安全系数降至阈值以下")
    else:
        report_lines.append(f"- ✓ 在所有分析水深范围内，边坡保持稳定")
    
    # 保存报告
    report_file = output_dir / 'analysis_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"✓ 报告已保存: {report_file}")


def main():
    """主程序"""
    # 配置文件路径
    config_file = Path(__file__).parent / "configs" / "slope_config.json"
    
    # 如果配置文件不存在，创建默认配置
    if not config_file.exists():
        config_file.parent.mkdir(exist_ok=True)
        default_config = {
            "soil": {
                "cohesion": 15.0,
                "friction_angle": 30.0,
                "unit_weight": 18.0,
                "saturated_unit_weight": 20.0
            },
            "slope": {
                "angle": 26.57
            }
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print(f"✓ 创建默认配置: {config_file}")
    
    # 加载配置
    config = load_config(config_file)
    
    # 运行分析
    output_dir = Path("./output/slope_analysis")
    analyze_slope_stability(config, output_dir)
    
    print("\n✓ 分析完成！")


if __name__ == "__main__":
    main()
```

---

## 7. 常见问题 (FAQ)

### Q1: 如何判断某个功能应该放在基础库还是应用层？

**判断标准**:
- **基础库**: 通用的、可复用的、与具体应用无关的功能
  - 例: 数值求解算法、物理模型计算
- **应用层**: 特定场景的、组合式的、包含业务逻辑的功能
  - 例: 生成特定格式的报告、特定项目的配置

### Q2: 基础库中的默认参数值算不算硬编码？

**不算**，只要满足以下条件:
1. 默认值是物理常数或行业标准值
2. 用户可以通过参数覆盖默认值
3. 默认值在文档中明确说明

### Q3: 发现基础库的bug，但修改会影响已有应用怎么办？

**处理方案**:
1. 如果是明确的错误，直接修复并升级PATCH版本
2. 如果涉及接口变更，考虑：
   - 保留旧接口并标记为废弃 (deprecated)
   - 提供新接口
   - 在CHANGELOG中明确说明迁移方法
3. 如果是破坏性变更，升级MAJOR版本

### Q4: 测试代码要放在哪里？

**位置**:
- **单元测试**: `tests/unit/` - 测试基础库的单个函数/类
- **集成测试**: `tests/integration/` - 测试多个模块的协同工作
- **场景测试**: `tests/scenarios/` - 测试特定应用场景

---

## 8. 工具和自动化

### 8.1 代码格式化

使用`black`进行代码格式化：

```bash
# 格式化所有Python文件
black channel_stability/ examples/ tests/

# 检查但不修改
black --check channel_stability/
```

### 8.2 代码质量检查

使用`pylint`或`flake8`：

```bash
# Pylint检查
pylint channel_stability/

# Flake8检查
flake8 channel_stability/ --max-line-length=100
```

### 8.3 类型检查

使用`mypy`进行类型检查：

```bash
mypy channel_stability/ --strict
```

### 8.4 测试覆盖率

使用`pytest-cov`：

```bash
pytest tests/ --cov=channel_stability --cov-report=html
```

---

## 9. 总结

### 核心原则回顾

1. **分层清晰**: 基础库和应用层职责明确
2. **参数化设计**: 基础库无硬编码，一切可配置
3. **文档完善**: 接口文档、使用示例齐全
4. **测试充分**: 单元测试、集成测试覆盖
5. **错误处理**: 输入验证、异常处理完善
6. **持续改进**: Bug修复优先，持续优化

### 检查清单

开发前检查:
- [ ] 明确功能归属（基础库/应用层）
- [ ] 设计接口和参数
- [ ] 编写接口文档

开发中检查:
- [ ] 无硬编码参数
- [ ] 完整类型注解
- [ ] 输入验证
- [ ] 编写单元测试

开发后检查:
- [ ] 测试通过
- [ ] 文档更新
- [ ] CHANGELOG更新
- [ ] 代码审查

---

**文档维护者**: 开发团队  
**最后更新**: 2025-10-27  
**文档版本**: 1.0

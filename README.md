# 中国水库自动化水资源监测与预测系统

## 项目简介

本项目是一个全自动化的水资源监测与预测平台，集成多源遥感数据、气象预报信息和水文模型，实现对中国境内主要水库的实时监测和预测。

## 主要功能

### 1. 自动数据采集
- ✅ 每日自动更新Sentinel/Landsat/MODIS遥感影像
- ✅ 实时获取ECMWF/GFS气象预报数据
- ✅ 采集全国水文站网观测数据
- ✅ 支持断点续传和增量更新

### 2. 智能监测
- ✅ 基于遥感影像的水体识别与水位反演
- ✅ 多源数据融合和质量控制
- ✅ 水库蓄水量动态监测
- ✅ 异常检测和预警

### 3. 预测预报
- ✅ 短期(1-7天)径流预报
- ✅ 中长期(月-季节)水资源预测
- ✅ 集合预报和不确定性评估
- ✅ 气候变化情景模拟

### 4. 流域建模
- ✅ 基于HydroSIS的自动流域划分
- ✅ 多种水文模型(XAJ/SWAT/VIC/LSTM)
- ✅ 自动参数率定和验证
- ✅ 分布式并行计算

## 系统架构

```
├── data_acquisition/          # 数据采集模块
│   ├── remote_sensing/       # 遥感数据采集
│   │   ├── sentinel.py       # Sentinel卫星
│   │   ├── landsat.py        # Landsat卫星
│   │   └── modis.py          # MODIS卫星
│   ├── meteorology/          # 气象数据采集
│   │   ├── ecmwf.py         # ECMWF数据
│   │   ├── gfs.py           # GFS数据
│   │   └── cma.py           # 中国气象数据
│   └── scheduler.py          # 任务调度
│
├── hydrological_model/       # 水文模型模块
│   ├── hydrosis_wrapper.py  # HydroSIS集成
│   └── model_manager.py     # 模型管理
│
├── database/                 # 数据库模块
│   └── database.py          # PostgreSQL/InfluxDB操作
│
└── config.py                 # 系统配置

```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd water_resource_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件并配置API密钥：

```bash
# Sentinel数据
SENTINEL_USERNAME=your_username
SENTINEL_PASSWORD=your_password

# USGS数据
USGS_USERNAME=your_username
USGS_PASSWORD=your_password

# NASA Earthdata
NASA_USERNAME=your_username
NASA_PASSWORD=your_password

# ECMWF数据
ECMWF_API_KEY=your_api_key

# 中国气象数据
CMA_API_KEY=your_api_key

# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=water_resources
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

INFLUXDB_HOST=localhost
INFLUXDB_PORT=8086
INFLUXDB_DB=water_monitoring
INFLUXDB_USER=admin
INFLUXDB_PASSWORD=your_password

REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. 数据库初始化

```bash
# 启动PostgreSQL和InfluxDB
docker-compose up -d

# 初始化数据库表
python scripts/init_database.py
```

### 4. 运行示例

```python
from water_resource_system.data_acquisition.remote_sensing.sentinel import SentinelCollector
from water_resource_system.config import API_CONFIGS, CACHE_DIR
from datetime import datetime, timedelta

# 创建Sentinel数据采集器
collector = SentinelCollector(
    API_CONFIGS['sentinel'],
    CACHE_DIR / 'sentinel'
)

# 采集最近7天的数据
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

# 指定长江流域
region = {
    "west": 90.0,
    "south": 24.0,
    "east": 122.0,
    "north": 35.0
}

files = collector.collect(start_date, end_date, region)
print(f"下载了 {len(files)} 个文件")
```

### 5. 部署Airflow调度系统

```bash
# 初始化Airflow数据库
airflow db init

# 创建管理员用户
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 生成DAG文件
python scripts/create_airflow_dag.py

# 启动Airflow
airflow webserver -p 8080 &
airflow scheduler &
```

访问 http://localhost:8080 查看调度任务。

## 数据源说明

### 遥感数据
- **Sentinel-2**: 10m分辨率，5天重访，免费
- **Landsat 8/9**: 30m分辨率，16天重访，免费
- **MODIS**: 250m-1km分辨率，每日，免费

### 气象数据
- **ECMWF ERA5**: 0.25°分辨率，全球再分析数据
- **GFS**: 0.25°分辨率，10天预报
- **中国气象数据**: 高分辨率地面观测

### 地形数据
- **SRTM DEM**: 30m分辨率全球数字高程模型
- **ASTER GDEM**: 30m分辨率数字高程模型

## HydroSIS集成

本系统集成了[HydroSIS](https://github.com/leixiaohui-1974/HydroSIS)项目，用于流域划分和水文模型构建。

### 使用方法

```python
from water_resource_system.hydrological_model import HydroSISWrapper
from pathlib import Path

# 创建HydroSIS包装器
wrapper = HydroSISWrapper(
    hydrosis_path=Path("/path/to/HydroSIS"),
    workspace=Path("./workspace")
)

# 流域划分
watershed = wrapper.delineate_watershed(
    dem_path=Path("./data/dem.tif"),
    outlet_point=(114.3, 30.5),  # 经纬度
    threshold_area=1.0  # km²
)

# 构建水文模型
model = wrapper.build_hydrological_model(
    watershed_info=watershed,
    model_type="XAJ"  # 新安江模型
)

# 模型率定
calibrated = wrapper.calibrate_model(
    model_config=model,
    observed_data=observed_df,
    method="SCE-UA"
)

# 运行预报
forecast = wrapper.run_forecast(
    model_config=calibrated,
    current_state=current_state,
    forecast_forcing=forecast_df,
    ensemble_members=10
)
```

## 全国水库建模策略

系统采用分级建模策略：

### 一级分区（10个流域）
- 松辽河区、海河区、黄河区、淮河区、长江区
- 珠江区、东南诸河区、西南诸河区、内陆河区、西北诸河区

### 二级分区
按主要支流和水系细分

### 三级分区
按水库控制流域细分，建立单库模型

### 建模流程
1. 下载全国DEM、土地利用、土壤数据
2. 自动流域划分
3. 提取水文参数
4. 批量模型构建
5. 并行参数率定
6. 验证和部署

## 性能优化

### 数据采集优化
- 多线程并行下载
- 断点续传
- 增量更新
- 本地缓存

### 模型计算优化
- 分布式并行计算
- GPU加速
- 模型简化
- 结果缓存

## 监控和告警

系统提供多级监控：

### 数据监控
- 数据下载成功率
- 数据完整性检查
- 数据时效性监控

### 模型监控
- 模型运行状态
- 预报精度评估
- 异常值检测

### 业务监控
- 水位超警戒线告警
- 入库流量异常告警
- 极端事件预警

## API文档

系统提供RESTful API接口：

```bash
# 查询水库列表
GET /api/reservoirs?basin=长江区

# 查询水位数据
GET /api/water_level/{reservoir_id}?start=2025-01-01&end=2025-01-31

# 获取预报结果
GET /api/forecast/{reservoir_id}?horizon=7

# 查询流域信息
GET /api/basin/{basin_name}
```

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

- 项目主页: [GitHub链接]
- 问题反馈: [Issues链接]
- 邮箱: example@example.com

## 致谢

- [HydroSIS](https://github.com/leixiaohui-1974/HydroSIS) - 流域水文建模框架
- Sentinel Hub - 遥感数据API
- ECMWF - 气象数据支持
- 所有开源贡献者

## 更新日志

### v1.0.0 (2025-10-26)
- ✨ 初始版本发布
- ✅ 实现数据采集模块
- ✅ 集成HydroSIS
- ✅ 完成数据库设计
- ✅ 添加Airflow调度

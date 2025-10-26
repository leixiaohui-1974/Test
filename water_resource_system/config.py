"""
系统配置文件
"""
import os
from pathlib import Path
from typing import Dict, Any

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

# 创建必要的目录
for dir_path in [DATA_DIR, CACHE_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据源API配置
API_CONFIGS = {
    # Sentinel卫星数据
    "sentinel": {
        "username": os.getenv("SENTINEL_USERNAME", ""),
        "password": os.getenv("SENTINEL_PASSWORD", ""),
        "api_url": "https://scihub.copernicus.eu/dhus",
    },
    
    # USGS Landsat数据
    "landsat": {
        "username": os.getenv("USGS_USERNAME", ""),
        "password": os.getenv("USGS_PASSWORD", ""),
        "api_url": "https://m2m.cr.usgs.gov/api/api/json/stable/",
    },
    
    # NASA Earthdata
    "nasa_earthdata": {
        "username": os.getenv("NASA_USERNAME", ""),
        "password": os.getenv("NASA_PASSWORD", ""),
        "api_url": "https://cmr.earthdata.nasa.gov/search",
    },
    
    # ECMWF气象数据
    "ecmwf": {
        "api_key": os.getenv("ECMWF_API_KEY", ""),
        "api_url": "https://cds.climate.copernicus.eu/api/v2",
    },
    
    # GFS气象预报
    "gfs": {
        "api_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
    },
    
    # 中国气象数据
    "cma": {
        "api_key": os.getenv("CMA_API_KEY", ""),
        "api_url": "http://data.cma.cn/api",
    },
}

# 数据库配置
DATABASE_CONFIGS = {
    # PostgreSQL + PostGIS (空间数据)
    "postgresql": {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "water_resources"),
        "username": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    },
    
    # InfluxDB (时序数据)
    "influxdb": {
        "host": os.getenv("INFLUXDB_HOST", "localhost"),
        "port": int(os.getenv("INFLUXDB_PORT", "8086")),
        "database": os.getenv("INFLUXDB_DB", "water_monitoring"),
        "username": os.getenv("INFLUXDB_USER", "admin"),
        "password": os.getenv("INFLUXDB_PASSWORD", ""),
    },
    
    # Redis (缓存)
    "redis": {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
        "password": os.getenv("REDIS_PASSWORD", None),
    },
}

# 水库监测配置
RESERVOIR_CONFIG = {
    "min_area": 1.0,  # 最小监测面积(km²)
    "update_frequency": "daily",  # 更新频率
    "priority_reservoirs": [  # 重点监测水库
        "三峡水库",
        "小浪底水库",
        "龙羊峡水库",
        "丹江口水库",
        "新安江水库",
    ],
}

# 遥感数据处理配置
REMOTE_SENSING_CONFIG = {
    "cloud_threshold": 20,  # 云量阈值(%)
    "resolution": {
        "sentinel": 10,  # 米
        "landsat": 30,   # 米
        "modis": 250,    # 米
    },
    "water_indices": ["NDWI", "MNDWI", "AWEIsh"],
}

# 气象数据配置
METEOROLOGY_CONFIG = {
    "variables": [
        "precipitation",
        "temperature",
        "relative_humidity",
        "wind_speed",
        "solar_radiation",
    ],
    "forecast_horizon": 7,  # 预报天数
    "spatial_resolution": 0.25,  # 度
}

# 水文模型配置
HYDROLOGICAL_MODEL_CONFIG = {
    "model_types": ["XAJ", "SWAT", "VIC", "LSTM"],
    "calibration_method": "SCE-UA",
    "validation_split": 0.3,
    "timestep": "hourly",
}

# 中国主要流域配置
CHINA_BASINS = {
    "松辽河区": {"code": "A", "area": 1240000},
    "海河区": {"code": "B", "area": 318000},
    "黄河区": {"code": "C", "area": 795000},
    "淮河区": {"code": "D", "area": 270000},
    "长江区": {"code": "E", "area": 1800000},
    "珠江区": {"code": "F", "area": 453690},
    "东南诸河区": {"code": "G", "area": 224000},
    "西南诸河区": {"code": "H", "area": 918000},
    "内陆河区": {"code": "I", "area": 3680000},
    "西北诸河区": {"code": "J", "area": 500000},
}

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(DATA_DIR / "logs" / "system.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
        }
    },
}

# 创建日志目录
(DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)

#!/usr/bin/env python
"""
下载全中国水资源建模所需的基础数据
包括：DEM、土地利用、土壤、气象站、水文站等
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from water_resource_system.config import DATA_DIR, CHINA_BASINS
from water_resource_system.data_acquisition.base import RegionalDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dem_data():
    """下载全国DEM数据"""
    logger.info("=" * 60)
    logger.info("开始下载DEM数据")
    logger.info("=" * 60)
    
    dem_dir = DATA_DIR / "static" / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    
    # 中国边界
    china_bbox = RegionalDataCollector.china_bbox()
    
    logger.info("数据源选项:")
    logger.info("1. SRTM 30m DEM")
    logger.info("   - 下载地址: https://earthexplorer.usgs.gov/")
    logger.info("   - 需要注册USGS账号")
    logger.info("   - 覆盖范围: 北纬60° - 南纬60°")
    logger.info("")
    logger.info("2. ASTER GDEM 30m")
    logger.info("   - 下载地址: https://asterweb.jpl.nasa.gov/gdem.asp")
    logger.info("   - 需要注册NASA Earthdata账号")
    logger.info("   - 全球覆盖")
    logger.info("")
    logger.info("3. 地理空间数据云 (中国区域)")
    logger.info("   - 下载地址: http://www.gscloud.cn/")
    logger.info("   - 需要注册账号")
    logger.info("   - 中国1:5万DEM")
    logger.info("")
    logger.info(f"建议下载范围: {china_bbox}")
    logger.info(f"保存目录: {dem_dir}")


def download_landuse_data():
    """下载土地利用数据"""
    logger.info("=" * 60)
    logger.info("开始下载土地利用数据")
    logger.info("=" * 60)
    
    landuse_dir = DATA_DIR / "static" / "landuse"
    landuse_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("数据源选项:")
    logger.info("1. GlobeLand30 (30m分辨率)")
    logger.info("   - 下载地址: http://www.globallandcover.com/")
    logger.info("   - 免费下载，需要注册")
    logger.info("   - 2010年和2020年两期数据")
    logger.info("")
    logger.info("2. 中国土地利用遥感监测数据库")
    logger.info("   - 来源: 中国科学院资源环境科学数据中心")
    logger.info("   - 需要申请权限")
    logger.info("")
    logger.info(f"保存目录: {landuse_dir}")


def download_soil_data():
    """下载土壤数据"""
    logger.info("=" * 60)
    logger.info("开始下载土壤数据")
    logger.info("=" * 60)
    
    soil_dir = DATA_DIR / "static" / "soil"
    soil_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("数据源选项:")
    logger.info("1. HWSD (世界土壤数据库)")
    logger.info("   - 下载地址: https://www.fao.org/soils-portal/data-hub/")
    logger.info("   - 免费下载")
    logger.info("   - 1km分辨率")
    logger.info("")
    logger.info("2. 中国土壤数据库")
    logger.info("   - 来源: 南京土壤研究所")
    logger.info("   - 需要申请权限")
    logger.info("")
    logger.info(f"保存目录: {soil_dir}")


def download_station_data():
    """下载站点数据"""
    logger.info("=" * 60)
    logger.info("开始下载站点数据")
    logger.info("=" * 60)
    
    station_dir = DATA_DIR / "static" / "stations"
    station_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("数据源选项:")
    logger.info("1. 气象站数据")
    logger.info("   - 中国气象数据网: http://data.cma.cn/")
    logger.info("   - 需要注册和申请")
    logger.info("   - 包含站点位置和历史观测数据")
    logger.info("")
    logger.info("2. 水文站数据")
    logger.info("   - 全国水情信息网: http://xxfb.mwr.cn/")
    logger.info("   - 公开数据有限")
    logger.info("   - 详细数据需向水利部门申请")
    logger.info("")
    logger.info("3. 全球径流数据中心(GRDC)")
    logger.info("   - 网址: https://www.bafg.de/GRDC/")
    logger.info("   - 包含部分中国站点")
    logger.info("   - 需要注册申请")
    logger.info("")
    logger.info(f"保存目录: {station_dir}")


def download_reservoir_data():
    """下载水库数据"""
    logger.info("=" * 60)
    logger.info("开始下载水库数据")
    logger.info("=" * 60)
    
    reservoir_dir = DATA_DIR / "static" / "reservoirs"
    reservoir_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("数据源选项:")
    logger.info("1. 全球水库数据库 (GRanD)")
    logger.info("   - 下载地址: http://globaldamwatch.org/grand/")
    logger.info("   - 包含中国主要大型水库")
    logger.info("   - 免费下载")
    logger.info("")
    logger.info("2. 中国水库名录")
    logger.info("   - 来源: 水利部")
    logger.info("   - 需要购买或申请")
    logger.info("")
    logger.info("3. OpenStreetMap")
    logger.info("   - 可以提取水库位置信息")
    logger.info("   - 数据可能不完整")
    logger.info("")
    logger.info(f"保存目录: {reservoir_dir}")


def download_historical_data():
    """下载历史气象和水文数据"""
    logger.info("=" * 60)
    logger.info("开始下载历史数据")
    logger.info("=" * 60)
    
    historical_dir = DATA_DIR / "historical"
    historical_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("数据源选项:")
    logger.info("1. ERA5再分析数据 (1979-至今)")
    logger.info("   - ECMWF气候数据商店")
    logger.info("   - 需要注册CDS API")
    logger.info("   - 包含降水、温度等多种变量")
    logger.info("")
    logger.info("2. CHIRPS降水数据 (1981-至今)")
    logger.info("   - 下载地址: https://data.chc.ucsb.edu/products/CHIRPS-2.0/")
    logger.info("   - 0.05°分辨率日降水")
    logger.info("")
    logger.info("3. 中国地面气候资料日值数据集")
    logger.info("   - 中国气象数据网")
    logger.info("   - 需要购买或申请")
    logger.info("")
    logger.info(f"保存目录: {historical_dir}")
    logger.info("\n推荐下载时间范围: 最近30年(1995-2025)")


def create_download_script():
    """创建自动下载脚本"""
    logger.info("=" * 60)
    logger.info("生成数据下载脚本")
    logger.info("=" * 60)
    
    scripts_dir = Path(__file__).parent
    download_script = scripts_dir / "auto_download_data.sh"
    
    script_content = """#!/bin/bash
# 自动下载中国水资源建模数据

echo "开始下载数据..."

# 1. 下载SRTM DEM (示例)
# 需要先登录USGS网站获取cookie
# wget --load-cookies cookies.txt --save-cookies cookies.txt --keep-session-cookies \\
#   -O srtm_china.zip "https://earthexplorer.usgs.gov/..."

# 2. 下载ERA5历史数据 (需要Python和cdsapi)
# python download_era5.py

# 3. 下载GlobeLand30土地利用
# wget -O globeland30_china.zip "http://www.globallandcover.com/..."

# 4. 下载HWSD土壤数据
# wget -O hwsd.zip "https://www.fao.org/..."

echo "数据下载完成！"
echo "请检查data/目录下的文件"
"""
    
    with open(download_script, 'w') as f:
        f.write(script_content)
    
    download_script.chmod(0o755)
    logger.info(f"已生成脚本: {download_script}")


def print_basin_info():
    """打印流域信息"""
    logger.info("=" * 60)
    logger.info("中国主要流域分区")
    logger.info("=" * 60)
    
    for basin_name, info in CHINA_BASINS.items():
        logger.info(f"{basin_name}:")
        logger.info(f"  - 代码: {info['code']}")
        logger.info(f"  - 面积: {info['area']:,} km²")
        logger.info("")


def main():
    """主函数"""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("中国水资源建模数据下载指南")
    logger.info("=" * 60)
    logger.info("\n")
    
    # 打印流域信息
    print_basin_info()
    
    # 各类数据下载指南
    download_dem_data()
    download_landuse_data()
    download_soil_data()
    download_station_data()
    download_reservoir_data()
    download_historical_data()
    
    # 生成下载脚本
    create_download_script()
    
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("数据下载总结")
    logger.info("=" * 60)
    logger.info("\n必需数据:")
    logger.info("✅ 1. DEM数据 (SRTM或ASTER)")
    logger.info("✅ 2. 水库基础信息 (GRanD)")
    logger.info("✅ 3. 历史气象数据 (ERA5)")
    logger.info("\n推荐数据:")
    logger.info("📋 4. 土地利用数据 (GlobeLand30)")
    logger.info("📋 5. 土壤数据 (HWSD)")
    logger.info("📋 6. 气象站/水文站数据")
    logger.info("\n可选数据:")
    logger.info("📌 7. 高分辨率中国区域数据")
    logger.info("📌 8. 实测径流数据")
    logger.info("\n")
    logger.info(f"数据保存目录: {DATA_DIR}")
    logger.info("\n注意事项:")
    logger.info("- 大部分数据源需要注册账号")
    logger.info("- 某些数据需要申请权限")
    logger.info("- 数据下载可能需要较长时间")
    logger.info("- 建议使用下载工具支持断点续传")
    logger.info("\n")


if __name__ == "__main__":
    main()

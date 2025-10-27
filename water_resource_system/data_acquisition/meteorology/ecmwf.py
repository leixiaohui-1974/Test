"""
ECMWF气象数据采集
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from ..base import RegionalDataCollector

logger = logging.getLogger(__name__)


class ECMWFCollector(RegionalDataCollector):
    """ECMWF气象数据采集器"""
    
    def __init__(self, config: Dict, cache_dir: Path):
        super().__init__(config, cache_dir)
        self.api_key = config.get("api_key")
        self.api_url = config.get("api_url")
        
    def collect(self, start_date: datetime, end_date: datetime, 
                region: Optional[Dict] = None) -> List[Path]:
        """
        采集ECMWF数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            region: 区域范围
            
        Returns:
            下载文件路径列表
        """
        try:
            import cdsapi
            
            # 创建CDS API客户端
            c = cdsapi.Client()
            
            # 如果没有指定区域，使用中国边界
            if region is None:
                region = self.china_bbox()
            
            # 转换为CDS API格式 [north, west, south, east]
            area = [
                region["north"],
                region["west"],
                region["south"],
                region["east"]
            ]
            
            downloaded_files = []
            
            # 按天下载数据
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                output_path = self.get_cache_path(
                    current_date, 
                    suffix="_ecmwf.nc"
                )
                
                if output_path.exists():
                    logger.info(f"数据已存在: {output_path}")
                    downloaded_files.append(output_path)
                    current_date += timedelta(days=1)
                    continue
                
                # 下载ERA5再分析数据
                try:
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': [
                                'total_precipitation',
                                '2m_temperature',
                                '2m_dewpoint_temperature',
                                '10m_u_component_of_wind',
                                '10m_v_component_of_wind',
                                'surface_pressure',
                                'surface_solar_radiation_downwards',
                            ],
                            'year': current_date.year,
                            'month': current_date.month,
                            'day': current_date.day,
                            'time': [
                                '00:00', '03:00', '06:00', '09:00',
                                '12:00', '15:00', '18:00', '21:00',
                            ],
                            'area': area,
                        },
                        str(output_path)
                    )
                    
                    downloaded_files.append(output_path)
                    logger.info(f"下载完成: {output_path}")
                    
                except Exception as e:
                    logger.error(f"下载失败: {date_str}, 错误: {e}")
                
                current_date += timedelta(days=1)
            
            return downloaded_files
            
        except ImportError:
            logger.error("cdsapi库未安装，请运行: pip install cdsapi")
            return []
        except Exception as e:
            logger.error(f"采集ECMWF数据失败: {e}")
            return []
    
    def collect_forecast(self, forecast_date: datetime, 
                        region: Optional[Dict] = None) -> Optional[Path]:
        """
        采集ECMWF预报数据
        
        Args:
            forecast_date: 预报日期
            region: 区域范围
            
        Returns:
            下载文件路径
        """
        try:
            import cdsapi
            
            c = cdsapi.Client()
            
            if region is None:
                region = self.china_bbox()
            
            area = [
                region["north"], region["west"],
                region["south"], region["east"]
            ]
            
            output_path = self.get_cache_path(
                forecast_date,
                suffix="_ecmwf_forecast.nc"
            )
            
            if output_path.exists():
                logger.info(f"预报数据已存在: {output_path}")
                return output_path
            
            # 下载HRES预报数据
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'ensemble_mean',
                    'format': 'netcdf',
                    'variable': [
                        'total_precipitation',
                        '2m_temperature',
                    ],
                    'year': forecast_date.year,
                    'month': forecast_date.month,
                    'day': forecast_date.day,
                    'leadtime_hour': list(range(0, 241, 6)),  # 10天预报
                    'area': area,
                },
                str(output_path)
            )
            
            logger.info(f"预报数据下载完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"采集ECMWF预报数据失败: {e}")
            return None
    
    def validate(self, file_path: Path) -> bool:
        """验证NetCDF文件完整性"""
        try:
            import netCDF4
            
            if not file_path.exists():
                return False
            
            # 尝试打开文件
            with netCDF4.Dataset(file_path, 'r') as nc:
                # 检查是否包含必要的变量
                required_vars = ['tp']  # total_precipitation
                for var in required_vars:
                    if var not in nc.variables:
                        logger.warning(f"缺少变量: {var}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证文件失败: {e}")
            return False
    
    def extract_precipitation(self, nc_path: Path, 
                            output_csv: Path) -> bool:
        """
        提取降水数据
        
        Args:
            nc_path: NetCDF文件路径
            output_csv: 输出CSV路径
            
        Returns:
            是否成功
        """
        try:
            import netCDF4
            import pandas as pd
            import numpy as np
            
            with netCDF4.Dataset(nc_path, 'r') as nc:
                # 读取坐标和数据
                lons = nc.variables['longitude'][:]
                lats = nc.variables['latitude'][:]
                times = nc.variables['time'][:]
                precip = nc.variables['tp'][:]  # m -> mm
                
                # 转换为DataFrame
                data_list = []
                for t_idx, time_val in enumerate(times):
                    for lat_idx, lat in enumerate(lats):
                        for lon_idx, lon in enumerate(lons):
                            value = precip[t_idx, lat_idx, lon_idx]
                            if not np.isnan(value):
                                data_list.append({
                                    'time': time_val,
                                    'latitude': lat,
                                    'longitude': lon,
                                    'precipitation_mm': value * 1000  # m转mm
                                })
                
                df = pd.DataFrame(data_list)
                df.to_csv(output_csv, index=False)
            
            logger.info(f"降水数据提取完成: {output_csv}")
            return True
            
        except Exception as e:
            logger.error(f"提取降水数据失败: {e}")
            return False

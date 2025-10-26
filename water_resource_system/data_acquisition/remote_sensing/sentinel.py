"""
Sentinel卫星数据采集
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from ..base import RegionalDataCollector

logger = logging.getLogger(__name__)


class SentinelCollector(RegionalDataCollector):
    """Sentinel-2卫星数据采集器"""
    
    def __init__(self, config: Dict, cache_dir: Path):
        super().__init__(config, cache_dir)
        self.api_url = config.get("api_url")
        self.username = config.get("username")
        self.password = config.get("password")
        
    def collect(self, start_date: datetime, end_date: datetime, 
                region: Optional[Dict] = None) -> List[Path]:
        """
        采集Sentinel-2数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            region: 区域范围
            
        Returns:
            下载文件路径列表
        """
        try:
            # 使用sentinelsat库
            from sentinelsat import SentinelAPI
            
            # 连接到API
            api = SentinelAPI(self.username, self.password, self.api_url)
            
            # 如果没有指定区域，使用中国边界
            if region is None:
                region = self.china_bbox()
            
            # 构建查询区域(WKT格式)
            footprint = self._bbox_to_wkt(region)
            
            # 查询产品
            products = api.query(
                footprint,
                date=(start_date, end_date),
                platformname='Sentinel-2',
                cloudcoverpercentage=(0, 20),  # 云量小于20%
                producttype='S2MSI2A'  # Level-2A产品
            )
            
            logger.info(f"找到 {len(products)} 个Sentinel-2产品")
            
            # 下载产品
            downloaded_files = []
            for product_id, product_info in products.items():
                # 检查是否已下载
                output_path = self.cache_dir / f"{product_info['title']}.zip"
                
                if output_path.exists():
                    logger.info(f"产品已存在: {output_path}")
                    downloaded_files.append(output_path)
                    continue
                
                # 下载产品
                try:
                    api.download(product_id, directory_path=str(self.cache_dir))
                    downloaded_files.append(output_path)
                    logger.info(f"下载完成: {output_path}")
                except Exception as e:
                    logger.error(f"下载失败: {product_id}, 错误: {e}")
            
            return downloaded_files
            
        except ImportError:
            logger.error("sentinelsat库未安装，请运行: pip install sentinelsat")
            return []
        except Exception as e:
            logger.error(f"采集Sentinel数据失败: {e}")
            return []
    
    def validate(self, file_path: Path) -> bool:
        """验证Sentinel产品完整性"""
        if not file_path.exists():
            return False
        
        # 检查文件大小(基本验证)
        file_size = file_path.stat().st_size
        if file_size < 1000000:  # 小于1MB可能不完整
            return False
        
        return True
    
    def extract_water_body(self, image_path: Path, output_path: Path,
                          method: str = "NDWI") -> bool:
        """
        提取水体
        
        Args:
            image_path: 输入影像路径
            output_path: 输出路径
            method: 提取方法 (NDWI, MNDWI, AWEIsh)
            
        Returns:
            是否成功
        """
        try:
            import rasterio
            import numpy as np
            
            # 读取影像
            with rasterio.open(image_path) as src:
                # 读取绿光和近红外波段
                green = src.read(3)  # B3
                nir = src.read(8)    # B8
                
                # 计算NDWI
                if method == "NDWI":
                    ndwi = (green.astype(float) - nir.astype(float)) / \
                           (green.astype(float) + nir.astype(float) + 1e-10)
                    
                    # 阈值分割
                    water_mask = ndwi > 0.3
                
                # 保存结果
                profile = src.profile
                profile.update(dtype=rasterio.uint8, count=1)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(water_mask.astype(np.uint8), 1)
            
            logger.info(f"水体提取完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"水体提取失败: {e}")
            return False
    
    @staticmethod
    def _bbox_to_wkt(bbox: Dict) -> str:
        """将边界框转换为WKT格式"""
        west, south = bbox["west"], bbox["south"]
        east, north = bbox["east"], bbox["north"]
        
        return (f"POLYGON(("
                f"{west} {north},"
                f"{east} {north},"
                f"{east} {south},"
                f"{west} {south},"
                f"{west} {north}))")

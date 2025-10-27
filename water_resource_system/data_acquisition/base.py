"""
数据采集基类
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BaseDataCollector(ABC):
    """数据采集器基类"""
    
    def __init__(self, config: Dict[str, Any], cache_dir: Path):
        """
        初始化数据采集器
        
        Args:
            config: API配置
            cache_dir: 缓存目录
        """
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建带重试机制的session
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """创建带重试机制的HTTP会话"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @abstractmethod
    def collect(self, start_date: datetime, end_date: datetime, 
                region: Optional[Dict] = None) -> List[Path]:
        """
        采集数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            region: 区域范围 (可选)
            
        Returns:
            下载文件路径列表
        """
        pass
    
    @abstractmethod
    def validate(self, file_path: Path) -> bool:
        """
        验证数据完整性
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否有效
        """
        pass
    
    def get_cache_path(self, date: datetime, suffix: str = "") -> Path:
        """
        获取缓存文件路径
        
        Args:
            date: 日期
            suffix: 文件后缀
            
        Returns:
            缓存路径
        """
        date_str = date.strftime("%Y%m%d")
        filename = f"{self.__class__.__name__}_{date_str}{suffix}"
        return self.cache_dir / filename
    
    def is_cached(self, date: datetime, suffix: str = "") -> bool:
        """检查数据是否已缓存"""
        cache_path = self.get_cache_path(date, suffix)
        return cache_path.exists()
    
    def download_file(self, url: str, output_path: Path, 
                     headers: Optional[Dict] = None,
                     timeout: int = 300) -> bool:
        """
        下载文件
        
        Args:
            url: 下载链接
            output_path: 输出路径
            headers: 请求头
            timeout: 超时时间(秒)
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"开始下载: {url}")
            
            response = self.session.get(
                url, 
                headers=headers,
                stream=True,
                timeout=timeout
            )
            response.raise_for_status()
            
            # 流式写入文件
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"下载完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {url}, 错误: {e}")
            return False
    
    def cleanup_old_cache(self, days: int = 30):
        """
        清理过期缓存
        
        Args:
            days: 保留天数
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        logger.info(f"删除过期缓存: {file_path}")
                    except Exception as e:
                        logger.error(f"删除缓存失败: {file_path}, 错误: {e}")


class RegionalDataCollector(BaseDataCollector):
    """区域数据采集器(支持空间范围查询)"""
    
    def __init__(self, config: Dict[str, Any], cache_dir: Path):
        super().__init__(config, cache_dir)
    
    def define_bbox(self, west: float, south: float, 
                   east: float, north: float) -> Dict:
        """
        定义边界框
        
        Args:
            west: 西经度
            south: 南纬度
            east: 东经度
            north: 北纬度
            
        Returns:
            边界框字典
        """
        return {
            "west": west,
            "south": south,
            "east": east,
            "north": north
        }
    
    @staticmethod
    def china_bbox() -> Dict:
        """获取中国边界框"""
        return {
            "west": 73.5,
            "south": 18.0,
            "east": 135.0,
            "north": 53.5
        }
    
    @staticmethod
    def basin_bbox(basin_name: str) -> Optional[Dict]:
        """
        获取流域边界框
        
        Args:
            basin_name: 流域名称
            
        Returns:
            边界框或None
        """
        # 主要流域边界框(示例数据)
        basins = {
            "长江区": {"west": 90.0, "south": 24.0, "east": 122.0, "north": 35.0},
            "黄河区": {"west": 96.0, "south": 32.0, "east": 119.0, "north": 42.0},
            "珠江区": {"west": 102.0, "south": 21.0, "east": 116.0, "north": 26.5},
            "海河区": {"west": 112.0, "south": 35.0, "east": 120.0, "north": 43.0},
            "淮河区": {"west": 111.0, "south": 31.0, "east": 121.0, "north": 36.0},
        }
        return basins.get(basin_name)

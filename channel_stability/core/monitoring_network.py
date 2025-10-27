"""
监测网络数据结构

定义地下水观测井、雨量计、水位计等监测设施
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum


class StationType(Enum):
    """监测站类型"""
    GROUNDWATER = "groundwater"  # 地下水观测井
    RAINFALL = "rainfall"  # 雨量计
    WATER_LEVEL = "water_level"  # 水位计
    COMBINED = "combined"  # 组合站


@dataclass
class MonitoringStation:
    """监测站"""
    station_id: str  # 站点ID
    station_type: StationType  # 站点类型
    station: float  # 桩号位置 (m)
    
    # 地理位置
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    
    # 观测数据（时间序列）
    observation_times: List[float] = field(default_factory=list)  # 观测时间 (s)
    
    # 地下水位数据
    groundwater_levels: List[float] = field(default_factory=list)  # 地下水位 (m)
    
    # 降雨数据
    rainfall_intensity: List[float] = field(default_factory=list)  # 降雨强度 (mm/h)
    cumulative_rainfall: List[float] = field(default_factory=list)  # 累积降雨 (mm)
    
    # 渠道水位数据
    water_levels: List[float] = field(default_factory=list)  # 水位 (m)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_observation(
        self,
        time: float,
        groundwater_level: Optional[float] = None,
        rainfall: Optional[float] = None,
        water_level: Optional[float] = None,
    ) -> None:
        """添加观测数据"""
        self.observation_times.append(time)
        
        if groundwater_level is not None:
            self.groundwater_levels.append(groundwater_level)
        if rainfall is not None:
            self.rainfall_intensity.append(rainfall)
        if water_level is not None:
            self.water_levels.append(water_level)
    
    def get_latest_observation(self) -> Tuple[float, Dict[str, float]]:
        """获取最新观测数据"""
        if not self.observation_times:
            return 0.0, {}
        
        latest_time = self.observation_times[-1]
        data = {}
        
        if self.groundwater_levels:
            data['groundwater_level'] = self.groundwater_levels[-1]
        if self.rainfall_intensity:
            data['rainfall_intensity'] = self.rainfall_intensity[-1]
        if self.water_levels:
            data['water_level'] = self.water_levels[-1]
        
        return latest_time, data
    
    def interpolate_groundwater_level(self, time: float) -> Optional[float]:
        """插值获取指定时刻的地下水位"""
        if not self.observation_times or not self.groundwater_levels:
            return None
        
        times = np.array(self.observation_times)
        levels = np.array(self.groundwater_levels)
        
        if time < times[0] or time > times[-1]:
            return None
        
        return np.interp(time, times, levels)


@dataclass
class MonitoringNetwork:
    """监测网络"""
    network_name: str
    stations: List[MonitoringStation] = field(default_factory=list)
    
    # 网络配置
    groundwater_spacing: float = 500.0  # 地下水观测井间距 (m)
    
    def __post_init__(self):
        """初始化后处理"""
        # 按桩号排序站点
        self.stations.sort(key=lambda s: s.station)
    
    @property
    def num_stations(self) -> int:
        """站点数量"""
        return len(self.stations)
    
    @property
    def groundwater_stations(self) -> List[MonitoringStation]:
        """获取所有地下水观测井"""
        return [
            s for s in self.stations
            if s.station_type in (StationType.GROUNDWATER, StationType.COMBINED)
        ]
    
    @property
    def rainfall_stations(self) -> List[MonitoringStation]:
        """获取所有雨量计"""
        return [
            s for s in self.stations
            if s.station_type in (StationType.RAINFALL, StationType.COMBINED)
        ]
    
    def get_station_by_id(self, station_id: str) -> Optional[MonitoringStation]:
        """根据ID获取站点"""
        for station in self.stations:
            if station.station_id == station_id:
                return station
        return None
    
    def get_nearest_station(
        self,
        station: float,
        station_type: Optional[StationType] = None,
    ) -> Optional[MonitoringStation]:
        """获取最近的监测站"""
        candidates = self.stations
        if station_type is not None:
            candidates = [s for s in self.stations if s.station_type == station_type]
        
        if not candidates:
            return None
        
        distances = [abs(s.station - station) for s in candidates]
        idx = np.argmin(distances)
        return candidates[idx]
    
    def create_uniform_groundwater_network(
        self,
        channel_length: float,
        spacing: float = 500.0,
    ) -> None:
        """
        创建均匀分布的地下水观测井网络
        
        Parameters
        ----------
        channel_length : float
            渠道长度 (m)
        spacing : float
            观测井间距 (m)
        """
        self.groundwater_spacing = spacing
        num_wells = int(channel_length / spacing) + 1
        
        for i in range(num_wells):
            station_loc = i * spacing
            if station_loc > channel_length:
                station_loc = channel_length
            
            station = MonitoringStation(
                station_id=f"GW_{i:03d}",
                station_type=StationType.GROUNDWATER,
                station=station_loc,
            )
            self.stations.append(station)
        
        self.stations.sort(key=lambda s: s.station)
    
    def add_boundary_stations(
        self,
        upstream_station: float,
        downstream_station: float,
    ) -> None:
        """
        在渠道两端添加组合监测站（雨量+水位）
        
        Parameters
        ----------
        upstream_station : float
            上游桩号
        downstream_station : float
            下游桩号
        """
        # 上游站
        upstream = MonitoringStation(
            station_id="UPSTREAM",
            station_type=StationType.COMBINED,
            station=upstream_station,
        )
        self.stations.append(upstream)
        
        # 下游站
        downstream = MonitoringStation(
            station_id="DOWNSTREAM",
            station_type=StationType.COMBINED,
            station=downstream_station,
        )
        self.stations.append(downstream)
        
        self.stations.sort(key=lambda s: s.station)
    
    def get_groundwater_profile(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定时刻的地下水位剖面
        
        Parameters
        ----------
        time : float
            时间 (s)
        
        Returns
        -------
        stations : np.ndarray
            桩号数组
        levels : np.ndarray
            地下水位数组
        """
        gw_stations = self.groundwater_stations
        if not gw_stations:
            return np.array([]), np.array([])
        
        stations = []
        levels = []
        
        for station in gw_stations:
            level = station.interpolate_groundwater_level(time)
            if level is not None:
                stations.append(station.station)
                levels.append(level)
        
        return np.array(stations), np.array(levels)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'network_name': self.network_name,
            'num_stations': self.num_stations,
            'groundwater_spacing': self.groundwater_spacing,
            'stations': [
                {
                    'station_id': s.station_id,
                    'station_type': s.station_type.value,
                    'station': s.station,
                    'latitude': s.latitude,
                    'longitude': s.longitude,
                    'elevation': s.elevation,
                }
                for s in self.stations
            ],
        }

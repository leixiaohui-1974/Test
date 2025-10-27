"""
明渠系统数据结构

定义明渠的几何、物理特性和监测网络
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class SoilProperties:
    """土壤物理特性"""
    cohesion: float  # 粘聚力 (kPa)
    friction_angle: float  # 内摩擦角 (度)
    unit_weight: float  # 容重 (kN/m³)
    saturated_unit_weight: float  # 饱和容重 (kN/m³)
    permeability: float  # 渗透系数 (m/s)
    porosity: float  # 孔隙率
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'cohesion': self.cohesion,
            'friction_angle': self.friction_angle,
            'unit_weight': self.unit_weight,
            'saturated_unit_weight': self.saturated_unit_weight,
            'permeability': self.permeability,
            'porosity': self.porosity,
        }


@dataclass
class LiningProperties:
    """衬砌板特性"""
    thickness: float  # 厚度 (m)
    density: float  # 密度 (kg/m³)
    elastic_modulus: float  # 弹性模量 (GPa)
    friction_coeff: float  # 与土体摩擦系数
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'thickness': self.thickness,
            'density': self.density,
            'elastic_modulus': self.elastic_modulus,
            'friction_coeff': self.friction_coeff,
        }


@dataclass
class ChannelSection:
    """明渠断面"""
    station: float  # 桩号 (m)
    bed_elevation: float  # 渠底高程 (m)
    bottom_width: float  # 渠底宽度 (m)
    side_slope: float  # 边坡坡度 (横:纵，如2.0表示1:2)
    max_depth: float  # 最大设计水深 (m)
    manning_n: float  # 曼宁糙率系数
    
    # 边坡特性
    slope_height: float  # 边坡高度 (m)
    slope_angle: float  # 边坡角度 (度)
    soil_properties: SoilProperties  # 土壤特性
    lining_properties: Optional[LiningProperties] = None  # 衬砌特性
    
    # 附加信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def area(self, depth: float) -> float:
        """计算过流断面积"""
        if depth <= 0:
            return 0.0
        return self.bottom_width * depth + self.side_slope * depth * depth
    
    def wetted_perimeter(self, depth: float) -> float:
        """计算湿周"""
        if depth <= 0:
            return 0.0
        return self.bottom_width + 2 * depth * np.sqrt(1 + self.side_slope**2)
    
    def hydraulic_radius(self, depth: float) -> float:
        """计算水力半径"""
        area = self.area(depth)
        wp = self.wetted_perimeter(depth)
        if wp <= 0:
            return 0.0
        return area / wp
    
    def top_width(self, depth: float) -> float:
        """计算水面宽度"""
        if depth <= 0:
            return self.bottom_width
        return self.bottom_width + 2 * self.side_slope * depth


@dataclass
class ChannelSystem:
    """明渠系统"""
    name: str
    total_length: float  # 总长度 (m)
    sections: List[ChannelSection]  # 断面列表
    
    # 边界条件参数
    upstream_station: float = 0.0
    downstream_station: Optional[float] = None
    
    # 元数据
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.downstream_station is None:
            self.downstream_station = self.total_length
        
        # 按桩号排序断面
        self.sections.sort(key=lambda s: s.station)
    
    @property
    def stations(self) -> np.ndarray:
        """获取所有断面桩号"""
        return np.array([s.station for s in self.sections])
    
    @property
    def num_sections(self) -> int:
        """断面数量"""
        return len(self.sections)
    
    def get_section_at(self, station: float) -> Optional[ChannelSection]:
        """获取指定桩号处的断面（最近邻）"""
        if not self.sections:
            return None
        
        stations = self.stations
        idx = np.argmin(np.abs(stations - station))
        return self.sections[idx]
    
    def interpolate_bed_elevation(self, station: float) -> float:
        """插值计算指定桩号处的渠底高程"""
        stations = self.stations
        elevations = np.array([s.bed_elevation for s in self.sections])
        return np.interp(station, stations, elevations)
    
    def create_uniform_sections(
        self,
        num_sections: int,
        base_section: ChannelSection,
        bed_slope: float = 0.0001,
    ) -> None:
        """
        创建均匀分布的断面
        
        Parameters
        ----------
        num_sections : int
            断面数量
        base_section : ChannelSection
            基准断面（用于复制几何和物理特性）
        bed_slope : float
            渠底纵坡
        """
        self.sections.clear()
        dx = self.total_length / (num_sections - 1)
        
        for i in range(num_sections):
            station = i * dx
            bed_elev = base_section.bed_elevation - bed_slope * station
            
            section = ChannelSection(
                station=station,
                bed_elevation=bed_elev,
                bottom_width=base_section.bottom_width,
                side_slope=base_section.side_slope,
                max_depth=base_section.max_depth,
                manning_n=base_section.manning_n,
                slope_height=base_section.slope_height,
                slope_angle=base_section.slope_angle,
                soil_properties=base_section.soil_properties,
                lining_properties=base_section.lining_properties,
            )
            self.sections.append(section)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'total_length': self.total_length,
            'num_sections': self.num_sections,
            'sections': [
                {
                    'station': s.station,
                    'bed_elevation': s.bed_elevation,
                    'bottom_width': s.bottom_width,
                    'side_slope': s.side_slope,
                    'max_depth': s.max_depth,
                    'manning_n': s.manning_n,
                    'slope_height': s.slope_height,
                    'slope_angle': s.slope_angle,
                    'soil_properties': s.soil_properties.to_dict(),
                    'lining_properties': s.lining_properties.to_dict() if s.lining_properties else None,
                }
                for s in self.sections
            ],
        }

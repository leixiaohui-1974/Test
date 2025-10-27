"""
边坡稳定性计算器

集成水动力学结果、地下水监测和降雨数据，计算边坡衬砌板稳定性
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..core.channel_system import ChannelSystem
from ..core.monitoring_network import MonitoringNetwork
from ..hydrodynamics.preissmann_solver import HydrodynamicResults
from .failure_mechanisms import LiningStabilityAnalysis, StabilityFactors


@dataclass
class StabilityResults:
    """边坡稳定性计算结果"""
    times: np.ndarray  # 时间序列
    stations: np.ndarray  # 断面桩号
    
    # 各断面的稳定性系数历史
    sliding_factors: np.ndarray  # (时间×断面)
    overturning_factors: np.ndarray
    uplift_factors: np.ndarray
    seepage_factors: np.ndarray
    comprehensive_factors: np.ndarray
    
    # 稳定性状态
    stability_status: np.ndarray  # (时间×断面), 1=稳定, 0=不稳定
    
    # 全渠道综合稳定性指标
    channel_stability_index: np.ndarray  # (时间,)
    unstable_section_count: np.ndarray  # (时间,), 不稳定断面数量
    
    def get_section_stability(self, station_index: int) -> Dict[str, np.ndarray]:
        """获取某断面的稳定性时间序列"""
        return {
            'times': self.times,
            'station': self.stations[station_index],
            'sliding_factor': self.sliding_factors[:, station_index],
            'overturning_factor': self.overturning_factors[:, station_index],
            'uplift_factor': self.uplift_factors[:, station_index],
            'seepage_factor': self.seepage_factors[:, station_index],
            'comprehensive_factor': self.comprehensive_factors[:, station_index],
            'is_stable': self.stability_status[:, station_index],
        }
    
    def get_time_snapshot(self, time_index: int) -> Dict[str, np.ndarray]:
        """获取某时刻的稳定性空间分布"""
        return {
            'time': self.times[time_index],
            'stations': self.stations,
            'sliding_factors': self.sliding_factors[time_index],
            'overturning_factors': self.overturning_factors[time_index],
            'uplift_factors': self.uplift_factors[time_index],
            'seepage_factors': self.seepage_factors[time_index],
            'comprehensive_factors': self.comprehensive_factors[time_index],
            'stability_status': self.stability_status[time_index],
        }


class SlopeStabilityCalculator:
    """边坡稳定性计算器"""
    
    def __init__(
        self,
        channel: ChannelSystem,
        monitoring_network: MonitoringNetwork,
    ):
        """
        初始化稳定性计算器
        
        Parameters
        ----------
        channel : ChannelSystem
            明渠系统
        monitoring_network : MonitoringNetwork
            监测网络
        """
        self.channel = channel
        self.monitoring_network = monitoring_network
    
    def interpolate_groundwater_level(
        self,
        station: float,
        time: float,
    ) -> float:
        """
        插值获取指定位置和时刻的地下水位
        
        Parameters
        ----------
        station : float
            桩号
        time : float
            时间
        
        Returns
        -------
        groundwater_level : float
            地下水位 (m，绝对高程)
        """
        # 获取所有地下水监测站
        gw_stations = self.monitoring_network.groundwater_stations
        
        if not gw_stations:
            # 如果没有监测数据，返回默认值
            return 0.0
        
        # 收集有效观测数据
        station_locs = []
        water_levels = []
        
        for gw_station in gw_stations:
            level = gw_station.interpolate_groundwater_level(time)
            if level is not None:
                station_locs.append(gw_station.station)
                water_levels.append(level)
        
        if not station_locs:
            return 0.0
        
        # 空间插值
        if len(station_locs) == 1:
            return water_levels[0]
        
        station_locs = np.array(station_locs)
        water_levels = np.array(water_levels)
        
        # 线性插值
        if station < station_locs[0]:
            return water_levels[0]
        elif station > station_locs[-1]:
            return water_levels[-1]
        else:
            return np.interp(station, station_locs, water_levels)
    
    def interpolate_rainfall(
        self,
        station: float,
        time: float,
    ) -> float:
        """
        插值获取指定位置和时刻的降雨强度
        
        Parameters
        ----------
        station : float
            桩号
        time : float
            时间
        
        Returns
        -------
        rainfall_intensity : float
            降雨强度 (mm/h)
        """
        # 获取雨量站
        rainfall_stations = self.monitoring_network.rainfall_stations
        
        if not rainfall_stations:
            return 0.0
        
        # 找到最近的雨量站
        nearest = self.monitoring_network.get_nearest_station(
            station, station_type=None
        )
        
        if nearest is None or not nearest.rainfall_intensity:
            return 0.0
        
        # 时间插值
        times = np.array(nearest.observation_times)
        intensities = np.array(nearest.rainfall_intensity)
        
        if time < times[0] or time > times[-1]:
            return 0.0
        
        return np.interp(time, times, intensities)
    
    def compute_stability(
        self,
        hydrodynamic_results: HydrodynamicResults,
        time_indices: Optional[List[int]] = None,
    ) -> StabilityResults:
        """
        计算边坡稳定性
        
        Parameters
        ----------
        hydrodynamic_results : HydrodynamicResults
            水动力学计算结果
        time_indices : List[int], optional
            需要计算的时间步索引，默认计算所有时间步
        
        Returns
        -------
        results : StabilityResults
            稳定性计算结果
        """
        times = hydrodynamic_results.times
        stations = hydrodynamic_results.stations
        water_levels = hydrodynamic_results.water_levels
        depths = hydrodynamic_results.depths
        
        if time_indices is None:
            time_indices = list(range(len(times)))
        
        num_times = len(time_indices)
        num_stations = len(stations)
        
        # 初始化结果数组
        sliding_factors = np.zeros((num_times, num_stations))
        overturning_factors = np.zeros((num_times, num_stations))
        uplift_factors = np.zeros((num_times, num_stations))
        seepage_factors = np.zeros((num_times, num_stations))
        comprehensive_factors = np.zeros((num_times, num_stations))
        stability_status = np.zeros((num_times, num_stations), dtype=int)
        
        # 对每个时间步和每个断面计算稳定性
        for t_idx, time_step in enumerate(time_indices):
            time = times[time_step]
            
            for s_idx, station in enumerate(stations):
                section = self.channel.sections[s_idx]
                
                # 获取该断面的渠道水位
                channel_water_level = depths[time_step, s_idx]
                
                # 获取地下水位
                groundwater_level = self.interpolate_groundwater_level(station, time)
                
                # 计算地下水埋深（从边坡顶部算起）
                slope_top_elevation = section.bed_elevation + section.slope_height
                groundwater_depth = slope_top_elevation - groundwater_level
                groundwater_depth = np.clip(groundwater_depth, 0.0, section.slope_height)
                
                # 获取降雨强度
                rainfall = self.interpolate_rainfall(station, time)
                
                # 如果该断面有衬砌
                if section.lining_properties is not None:
                    # 计算稳定性
                    factors = LiningStabilityAnalysis.compute_comprehensive_stability(
                        slope_angle=section.slope_angle,
                        slope_height=section.slope_height,
                        soil_props=section.soil_properties,
                        lining_props=section.lining_properties,
                        groundwater_depth=groundwater_depth,
                        channel_water_level=channel_water_level,
                        rainfall_intensity=rainfall,
                    )
                    
                    sliding_factors[t_idx, s_idx] = factors.sliding_factor
                    overturning_factors[t_idx, s_idx] = factors.overturning_factor
                    uplift_factors[t_idx, s_idx] = factors.uplift_factor
                    seepage_factors[t_idx, s_idx] = factors.seepage_factor
                    comprehensive_factors[t_idx, s_idx] = factors.comprehensive_factor
                    stability_status[t_idx, s_idx] = 1 if factors.is_stable else 0
                else:
                    # 无衬砌，标记为稳定
                    sliding_factors[t_idx, s_idx] = 999.9
                    overturning_factors[t_idx, s_idx] = 999.9
                    uplift_factors[t_idx, s_idx] = 999.9
                    seepage_factors[t_idx, s_idx] = 999.9
                    comprehensive_factors[t_idx, s_idx] = 999.9
                    stability_status[t_idx, s_idx] = 1
        
        # 计算全渠道综合稳定性指标
        # 方法1：使用所有断面综合稳定系数的平均值
        channel_stability_index = np.mean(comprehensive_factors, axis=1)
        
        # 方法2：统计不稳定断面数量
        unstable_section_count = num_stations - np.sum(stability_status, axis=1)
        
        return StabilityResults(
            times=times[time_indices],
            stations=stations,
            sliding_factors=sliding_factors,
            overturning_factors=overturning_factors,
            uplift_factors=uplift_factors,
            seepage_factors=seepage_factors,
            comprehensive_factors=comprehensive_factors,
            stability_status=stability_status,
            channel_stability_index=channel_stability_index,
            unstable_section_count=unstable_section_count,
        )

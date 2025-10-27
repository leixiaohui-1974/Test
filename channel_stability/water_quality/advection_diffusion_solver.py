"""
对流-扩散-反应方程求解器

用于水质模拟的一维数值求解器
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..hydrodynamics.preissmann_solver import HydrodynamicResults
from .reaction_kinetics import ReactionKinetics, WaterQualityParameters


@dataclass
class WaterQualityResults:
    """水质模拟结果"""
    times: np.ndarray  # 时间序列
    stations: np.ndarray  # 断面桩号
    concentrations: Dict[str, np.ndarray]  # 浓度场 {指标名: (时间×断面)}
    
    def get_concentration(self, indicator: str, time_index: int) -> np.ndarray:
        """获取某一时刻的浓度分布"""
        if indicator not in self.concentrations:
            raise ValueError(f"水质指标 {indicator} 不存在")
        return self.concentrations[indicator][time_index]
    
    def get_timeseries(self, indicator: str, station_index: int) -> np.ndarray:
        """获取某一断面的时间序列"""
        if indicator not in self.concentrations:
            raise ValueError(f"水质指标 {indicator} 不存在")
        return self.concentrations[indicator][:, station_index]


class WaterQualitySolver:
    """水质模拟求解器（基于对流-扩散-反应方程）"""
    
    def __init__(
        self,
        parameters: WaterQualityParameters,
        dispersion_coeff: float = 10.0,  # 纵向离散系数 (m²/s)
        upstream_concentrations: Optional[Dict[str, float]] = None,
        initial_concentrations: Optional[Dict[str, float]] = None,
    ):
        """
        初始化水质求解器
        
        Parameters
        ----------
        parameters : WaterQualityParameters
            水质参数
        dispersion_coeff : float
            纵向离散系数
        upstream_concentrations : Dict[str, float], optional
            上游边界浓度
        initial_concentrations : Dict[str, float], optional
            初始浓度场
        """
        self.params = parameters
        self.dispersion_coeff = dispersion_coeff
        
        # 默认上游边界浓度
        self.upstream_concentrations = upstream_concentrations or {
            'DO': 9.0,
            'BOD': 5.0,
            'COD': 20.0,
            'NH3N': 1.0,
            'TN': 5.0,
            'TP': 0.5,
            'SS': 10.0,
        }
        
        # 默认初始浓度
        self.initial_concentrations = initial_concentrations or self.upstream_concentrations.copy()
    
    def solve(
        self,
        hydrodynamic_results: HydrodynamicResults,
        indicators: List[str],
        dt: Optional[float] = None,
    ) -> WaterQualityResults:
        """
        求解水质方程
        
        Parameters
        ----------
        hydrodynamic_results : HydrodynamicResults
            水动力学计算结果
        indicators : List[str]
            需要模拟的水质指标列表
        dt : float, optional
            时间步长，默认使用水动力学结果的时间步长
        
        Returns
        -------
        results : WaterQualityResults
            水质模拟结果
        """
        times = hydrodynamic_results.times
        stations = hydrodynamic_results.stations
        depths = hydrodynamic_results.depths
        discharges = hydrodynamic_results.discharges
        velocities = hydrodynamic_results.velocities
        
        num_times = len(times)
        num_stations = len(stations)
        
        if dt is None:
            dt = times[1] - times[0] if len(times) > 1 else 1.0
        
        # 初始化浓度场
        concentrations = {}
        for indicator in indicators:
            initial_value = self.initial_concentrations.get(indicator, 0.0)
            concentrations[indicator] = np.full((num_times, num_stations), initial_value, dtype=float)
        
        # 空间步长
        dx = np.diff(stations)
        dx_avg = np.mean(dx)
        
        # 时间步进
        for t_idx in range(1, num_times):
            # 当前时刻的水动力学状态
            current_depths = depths[t_idx]
            current_velocities = velocities[t_idx]
            current_discharges = discharges[t_idx]
            
            # 对每个水质指标求解
            for indicator in indicators:
                C_old = concentrations[indicator][t_idx - 1].copy()
                C_new = C_old.copy()
                
                # 上游边界条件
                C_new[0] = self.upstream_concentrations.get(indicator, C_old[0])
                
                # 内部节点（使用一阶迎风+中心差分格式）
                for i in range(1, num_stations - 1):
                    depth = current_depths[i]
                    velocity = current_velocities[i]
                    
                    if depth < 1e-3:
                        C_new[i] = C_old[i]
                        continue
                    
                    # 对流项（一阶迎风格式）
                    if velocity > 0:
                        dC_dx = (C_old[i] - C_old[i - 1]) / dx_avg
                    else:
                        dC_dx = (C_old[i + 1] - C_old[i]) / dx_avg if i < num_stations - 1 else 0.0
                    
                    advection = -velocity * dC_dx
                    
                    # 扩散项（中心差分）
                    d2C_dx2 = (C_old[i + 1] - 2 * C_old[i] + C_old[i - 1]) / (dx_avg ** 2)
                    diffusion = self.dispersion_coeff * d2C_dx2
                    
                    # 反应源项
                    current_conc = {ind: concentrations[ind][t_idx - 1, i] for ind in indicators}
                    sources = ReactionKinetics.compute_all_sources(
                        current_conc, velocity, self.params
                    )
                    reaction = sources.get(indicator, 0.0) / 86400.0  # 转换为 1/s
                    
                    # 更新浓度
                    dC_dt = advection + diffusion + reaction
                    C_new[i] = C_old[i] + dt * dC_dt
                    
                    # 保证非负
                    C_new[i] = max(C_new[i], 0.0)
                
                # 下游边界条件（自由边界，梯度为零）
                C_new[-1] = C_new[-2]
                
                # 存储结果
                concentrations[indicator][t_idx] = C_new
        
        return WaterQualityResults(
            times=times,
            stations=stations,
            concentrations=concentrations,
        )
    
    def set_upstream_concentration(self, indicator: str, concentration: float) -> None:
        """设置上游边界浓度"""
        self.upstream_concentrations[indicator] = concentration
    
    def set_initial_concentration(self, indicator: str, concentration: float) -> None:
        """设置初始浓度"""
        self.initial_concentrations[indicator] = concentration

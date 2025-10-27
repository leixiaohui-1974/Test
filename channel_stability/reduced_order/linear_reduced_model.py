"""
线性化降阶模型

基于线性化理论的快速求解方法
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags, linalg as sp_linalg


@dataclass
class LinearizedState:
    """线性化状态"""
    times: np.ndarray
    stations: np.ndarray
    values: np.ndarray  # (time × station)


class LinearizedHydrodynamics:
    """
    线性化水动力学模型（ID模型）
    
    基于线性化圣维南方程的瞬态演进模型（Instantaneous Discharge Model）
    """
    
    def __init__(
        self,
        stations: np.ndarray,
        bed_elevations: np.ndarray,
        bottom_widths: np.ndarray,
        manning_n: float = 0.025,
        reference_depth: float = 2.0,
        reference_discharge: float = 10.0,
    ):
        """
        初始化线性化水动力学模型
        
        Parameters
        ----------
        stations : np.ndarray
            断面桩号
        bed_elevations : np.ndarray
            渠底高程
        bottom_widths : np.ndarray
            渠底宽度
        manning_n : float
            曼宁糙率
        reference_depth : float
            参考水深（线性化点）
        reference_discharge : float
            参考流量（线性化点）
        """
        self.stations = stations
        self.bed_elevations = bed_elevations
        self.bottom_widths = bottom_widths
        self.manning_n = manning_n
        self.reference_depth = reference_depth
        self.reference_discharge = reference_discharge
        
        self.num_sections = len(stations)
        self.dx = np.diff(stations)
        
        # 计算参考状态的水力参数
        self._compute_reference_parameters()
    
    def _compute_reference_parameters(self) -> None:
        """计算参考状态的水力参数"""
        h0 = self.reference_depth
        Q0 = self.reference_discharge
        
        # 参考面积
        self.A0 = self.bottom_widths * h0
        
        # 参考流速
        self.U0 = Q0 / self.A0
        
        # 参考波速（简化，使用矩形断面）
        g = 9.81
        self.C0 = np.sqrt(g * h0)
        
        # 摩阻系数（线性化）
        R0 = self.A0 / (self.bottom_widths + 2 * h0)  # 水力半径
        self.friction_coeff = (
            2 * g * self.manning_n ** 2 * np.abs(Q0) / (self.A0 * R0 ** (4.0 / 3.0))
        )
    
    def solve_kinematic_wave(
        self,
        total_time: float,
        dt: float,
        upstream_discharge_series: np.ndarray,  # (time,)
        initial_depth: Optional[np.ndarray] = None,
    ) -> LinearizedState:
        """
        运动波近似求解（ID模型）
        
        假设：局部惯性项可忽略，水深由流量唯一确定
        
        Parameters
        ----------
        total_time : float
            总时间
        dt : float
            时间步长
        upstream_discharge_series : np.ndarray
            上游流量过程
        initial_depth : np.ndarray, optional
            初始水深
        
        Returns
        -------
        state : LinearizedState
            水深状态
        """
        times = np.arange(0, total_time + dt, dt)
        num_times = len(times)
        
        # 初始化水深
        if initial_depth is None:
            depths = np.full((num_times, self.num_sections), self.reference_depth)
        else:
            depths = np.zeros((num_times, self.num_sections))
            depths[0] = initial_depth
        
        # 确保上游流量序列长度匹配
        if len(upstream_discharge_series) < num_times:
            # 补齐
            Q_up = np.zeros(num_times)
            Q_up[:len(upstream_discharge_series)] = upstream_discharge_series
            Q_up[len(upstream_discharge_series):] = upstream_discharge_series[-1]
        else:
            Q_up = upstream_discharge_series[:num_times]
        
        # 波速（简化为常数）
        wave_speed = self.C0[0] + self.U0[0]
        
        # 时间步进（简化的运动波传播）
        for t_idx in range(1, num_times):
            # 上游边界
            depths[t_idx, 0] = self._discharge_to_depth(Q_up[t_idx], 0)
            
            # 向下游传播
            for i in range(1, self.num_sections):
                # 时间延迟
                travel_time = self.dx[i - 1] / wave_speed
                delay_steps = int(travel_time / dt)
                
                if t_idx > delay_steps:
                    depths[t_idx, i] = depths[t_idx - delay_steps, i - 1]
                else:
                    depths[t_idx, i] = depths[t_idx - 1, i]
                
                # 添加摩阻衰减
                friction_decay = np.exp(-self.friction_coeff[i] * self.dx[i - 1] / wave_speed)
                depths[t_idx, i] *= friction_decay
        
        return LinearizedState(
            times=times,
            stations=self.stations,
            values=depths,
        )
    
    def _discharge_to_depth(self, discharge: float, section_idx: int) -> float:
        """根据流量计算水深（Manning公式反解）"""
        if discharge <= 0:
            return 1e-3
        
        b = self.bottom_widths[section_idx]
        n = self.manning_n
        S0 = 0.0001  # 假设底坡
        
        # 简化：矩形断面，迭代求解
        h = self.reference_depth
        for _ in range(5):
            A = b * h
            P = b + 2 * h
            R = A / P if P > 0 else 0
            Q_calc = (A / n) * (R ** (2.0 / 3.0)) * (S0 ** 0.5)
            
            if abs(Q_calc - discharge) < 0.01:
                break
            
            # 牛顿迭代
            dQ_dh = (b / n) * (R ** (2.0 / 3.0)) * (S0 ** 0.5)
            h = h + (discharge - Q_calc) / (dQ_dh + 1e-6)
            h = max(h, 1e-3)
        
        return h


class LinearizedWaterQuality:
    """线性化水质模型"""
    
    def __init__(
        self,
        stations: np.ndarray,
        reference_velocity: float = 0.5,
        dispersion_coeff: float = 10.0,
        decay_rate: float = 0.1,  # 一阶衰减率 (1/day)
    ):
        """
        初始化线性化水质模型
        
        Parameters
        ----------
        stations : np.ndarray
            断面桩号
        reference_velocity : float
            参考流速
        dispersion_coeff : float
            离散系数
        decay_rate : float
            一阶衰减率
        """
        self.stations = stations
        self.reference_velocity = reference_velocity
        self.dispersion_coeff = dispersion_coeff
        self.decay_rate = decay_rate
        
        self.num_sections = len(stations)
        self.dx = np.mean(np.diff(stations))
    
    def solve_linear_advection_diffusion(
        self,
        total_time: float,
        dt: float,
        upstream_concentration: float,
        initial_concentration: Optional[np.ndarray] = None,
    ) -> LinearizedState:
        """
        求解线性对流-扩散-衰减方程
        
        使用隐式有限差分法
        
        Parameters
        ----------
        total_time : float
            总时间
        dt : float
            时间步长
        upstream_concentration : float
            上游边界浓度
        initial_concentration : np.ndarray, optional
            初始浓度
        
        Returns
        -------
        state : LinearizedState
            浓度状态
        """
        times = np.arange(0, total_time + dt, dt)
        num_times = len(times)
        
        # 初始化浓度
        if initial_concentration is None:
            C = np.full((num_times, self.num_sections), upstream_concentration)
        else:
            C = np.zeros((num_times, self.num_sections))
            C[0] = initial_concentration
        
        # 无量纲参数
        Pe = self.reference_velocity * self.dx / self.dispersion_coeff  # Peclet数
        Cr = self.reference_velocity * dt / self.dx  # Courant数
        Da = self.decay_rate * dt / 86400.0  # Damköhler数（转换为s）
        
        # 构造三对角矩阵（隐式格式）
        # C_new[i] = C_old[i] + dt * (advection + diffusion + reaction)
        
        # 对流项系数
        a_adv = -Cr / 2.0
        c_adv = Cr / 2.0
        
        # 扩散项系数
        a_diff = -1.0 / (Pe * self.dx)
        b_diff = 2.0 / (Pe * self.dx)
        c_diff = -1.0 / (Pe * self.dx)
        
        # 总系数
        a = a_adv + a_diff
        b = 1.0 + b_diff + Da
        c = c_adv + c_diff
        
        # 构造三对角矩阵
        diagonals = [
            np.full(self.num_sections - 1, a),
            np.full(self.num_sections, b),
            np.full(self.num_sections - 1, c),
        ]
        offsets = [-1, 0, 1]
        A_matrix = diags(diagonals, offsets, format='csr')
        
        # 时间步进
        for t_idx in range(1, num_times):
            rhs = C[t_idx - 1].copy()
            
            # 上游边界
            rhs[0] = upstream_concentration
            
            # 下游边界（自由边界）
            # 不需要特殊处理
            
            # 求解线性方程组
            C[t_idx] = sp_linalg.spsolve(A_matrix, rhs)
            
            # 保证非负
            C[t_idx] = np.maximum(C[t_idx], 0.0)
        
        return LinearizedState(
            times=times,
            stations=self.stations,
            values=C,
        )


class LinearizedSlopeStability:
    """线性化边坡稳定性模型"""
    
    def __init__(
        self,
        stations: np.ndarray,
        base_stability_factor: float = 2.0,
        groundwater_sensitivity: float = -0.5,  # 地下水位上升对稳定性的影响
        channel_water_sensitivity: float = 0.2,  # 渠道水位上升对稳定性的影响
        rainfall_sensitivity: float = -0.1,  # 降雨对稳定性的影响
    ):
        """
        初始化线性化边坡稳定性模型
        
        基于线性响应假设：
        F = F0 + k_gw * Δh_gw + k_ch * Δh_ch + k_rain * ΔR
        
        Parameters
        ----------
        stations : np.ndarray
            断面桩号
        base_stability_factor : float
            基准稳定系数
        groundwater_sensitivity : float
            地下水位敏感系数
        channel_water_sensitivity : float
            渠道水位敏感系数
        rainfall_sensitivity : float
            降雨敏感系数
        """
        self.stations = stations
        self.F0 = base_stability_factor
        self.k_gw = groundwater_sensitivity
        self.k_ch = channel_water_sensitivity
        self.k_rain = rainfall_sensitivity
        
        self.num_sections = len(stations)
        
        # 参考状态
        self.reference_gw_level = 0.0
        self.reference_ch_level = 2.0
        self.reference_rainfall = 0.0
    
    def compute_stability(
        self,
        groundwater_levels: np.ndarray,  # (time × station)
        channel_water_levels: np.ndarray,  # (time × station)
        rainfall_intensities: np.ndarray,  # (time × station)
        times: np.ndarray,
    ) -> LinearizedState:
        """
        计算稳定性系数
        
        Parameters
        ----------
        groundwater_levels : np.ndarray
            地下水位场
        channel_water_levels : np.ndarray
            渠道水位场
        rainfall_intensities : np.ndarray
            降雨强度场
        times : np.ndarray
            时间序列
        
        Returns
        -------
        state : LinearizedState
            稳定性系数
        """
        num_times = len(times)
        
        # 计算偏差
        delta_gw = groundwater_levels - self.reference_gw_level
        delta_ch = channel_water_levels - self.reference_ch_level
        delta_rain = rainfall_intensities - self.reference_rainfall
        
        # 线性响应模型
        stability_factors = (
            self.F0 +
            self.k_gw * delta_gw +
            self.k_ch * delta_ch +
            self.k_rain * delta_rain
        )
        
        # 保证稳定系数在合理范围内
        stability_factors = np.clip(stability_factors, 0.5, 10.0)
        
        return LinearizedState(
            times=times,
            stations=self.stations,
            values=stability_factors,
        )
    
    def calibrate(
        self,
        groundwater_levels: np.ndarray,
        channel_water_levels: np.ndarray,
        rainfall_intensities: np.ndarray,
        observed_stability: np.ndarray,
    ) -> Dict[str, float]:
        """
        校准敏感系数
        
        使用最小二乘法拟合观测数据
        
        Parameters
        ----------
        groundwater_levels : np.ndarray
            地下水位观测值
        channel_water_levels : np.ndarray
            渠道水位观测值
        rainfall_intensities : np.ndarray
            降雨强度观测值
        observed_stability : np.ndarray
            观测的稳定系数
        
        Returns
        -------
        coefficients : Dict[str, float]
            校准后的系数
        """
        # 构造设计矩阵
        n_samples = len(groundwater_levels)
        X = np.column_stack([
            np.ones(n_samples),
            groundwater_levels - self.reference_gw_level,
            channel_water_levels - self.reference_ch_level,
            rainfall_intensities - self.reference_rainfall,
        ])
        
        # 最小二乘拟合
        coeffs = np.linalg.lstsq(X, observed_stability, rcond=None)[0]
        
        self.F0 = coeffs[0]
        self.k_gw = coeffs[1]
        self.k_ch = coeffs[2]
        self.k_rain = coeffs[3]
        
        return {
            'base_stability': self.F0,
            'groundwater_sensitivity': self.k_gw,
            'channel_water_sensitivity': self.k_ch,
            'rainfall_sensitivity': self.k_rain,
        }

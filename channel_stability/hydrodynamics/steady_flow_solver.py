"""
恒定流求解器

用于计算恒定流初值，为非恒定流模拟提供初始条件
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from ..core.channel_system import ChannelSystem


class SteadyFlowSolver:
    """恒定流求解器 - 使用标准步长法"""
    
    def __init__(
        self,
        channel: ChannelSystem,
        min_depth: float = 1e-3,
        max_iterations: int = 50,
        convergence_tol: float = 1e-4,
    ):
        """
        初始化恒定流求解器
        
        Parameters
        ----------
        channel : ChannelSystem
            明渠系统
        min_depth : float
            最小水深 (m)
        max_iterations : int
            最大迭代次数
        convergence_tol : float
            收敛容差 (m)
        """
        self.channel = channel
        self.min_depth = min_depth
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
    
    def solve_gradually_varied_flow(
        self,
        discharge: float,
        downstream_depth: float,
    ) -> Dict[str, np.ndarray]:
        """
        求解渐变流（从下游向上游计算）
        
        Parameters
        ----------
        discharge : float
            恒定流量 (m³/s)
        downstream_depth : float
            下游边界水深 (m)
        
        Returns
        -------
        results : Dict[str, np.ndarray]
            包含 stations, depths, velocities, water_levels, froude_numbers
        """
        sections = self.channel.sections
        stations = self.channel.stations
        num_sections = len(sections)
        
        depths = np.zeros(num_sections)
        velocities = np.zeros(num_sections)
        water_levels = np.zeros(num_sections)
        froude_numbers = np.zeros(num_sections)
        
        # 下游边界条件
        depths[-1] = max(downstream_depth, self.min_depth)
        
        # 从下游向上游推进
        for i in range(num_sections - 1, 0, -1):
            section_up = sections[i - 1]
            section_down = sections[i]
            
            dx = stations[i] - stations[i - 1]
            h_down = depths[i]
            
            # 标准步长法迭代
            h_up = h_down  # 初始猜测
            
            for iteration in range(self.max_iterations):
                h_old = h_up
                
                # 上游断面水力特性
                A_up = section_up.area(h_up)
                R_up = section_up.hydraulic_radius(h_up)
                V_up = discharge / A_up if A_up > 1e-6 else 0.0
                B_up = section_up.top_width(h_up)
                
                # 下游断面水力特性
                A_down = section_down.area(h_down)
                V_down = discharge / A_down if A_down > 1e-6 else 0.0
                B_down = section_down.top_width(h_down)
                
                # 能量方程
                # H = z + h + V²/(2g)
                z_up = section_up.bed_elevation
                z_down = section_down.bed_elevation
                
                H_up = z_up + h_up + V_up**2 / (2 * 9.81)
                H_down = z_down + h_down + V_down**2 / (2 * 9.81)
                
                # 平均摩阻坡度
                Sf_up = self._friction_slope(section_up, h_up, discharge)
                Sf_down = self._friction_slope(section_down, h_down, discharge)
                Sf_avg = (Sf_up + Sf_down) / 2.0
                
                # 能量损失
                hf = Sf_avg * dx
                
                # 能量方程： H_up = H_down + hf
                H_target = H_down + hf
                
                # 求解 h_up，使得 z_up + h_up + V_up²/(2g) = H_target
                # 这是一个非线性方程，使用牛顿法
                h_new = self._solve_depth_from_energy(
                    section_up, discharge, H_target
                )
                
                h_up = h_new
                
                # 检查收敛
                if abs(h_up - h_old) < self.convergence_tol:
                    break
            
            depths[i - 1] = h_up
        
        # 计算派生量
        for i, section in enumerate(sections):
            h = depths[i]
            A = section.area(h)
            V = discharge / A if A > 1e-6 else 0.0
            velocities[i] = V
            water_levels[i] = section.bed_elevation + h
            
            # Froude数
            if h > self.min_depth:
                B = section.top_width(h)
                if B > 1e-6:
                    Fr = V / np.sqrt(9.81 * A / B)
                else:
                    Fr = 0.0
            else:
                Fr = 0.0
            froude_numbers[i] = Fr
        
        return {
            'stations': stations,
            'depths': depths,
            'velocities': velocities,
            'water_levels': water_levels,
            'froude_numbers': froude_numbers,
            'discharge': discharge,
        }
    
    def _friction_slope(self, section, depth: float, discharge: float) -> float:
        """计算摩阻坡度（Manning公式）"""
        if depth <= self.min_depth:
            return 0.0
        
        A = section.area(depth)
        if A <= 1e-6:
            return 0.0
        
        R = section.hydraulic_radius(depth)
        if R <= 0:
            return 0.0
        
        n = section.manning_n
        V = discharge / A
        Sf = (n * V) ** 2 / (R ** (4.0 / 3.0))
        
        return max(Sf, 0.0)
    
    def _solve_depth_from_energy(
        self,
        section,
        discharge: float,
        target_energy: float,
    ) -> float:
        """
        从能量方程求解水深
        
        H = z + h + Q²/(2gA²) = target_energy
        """
        z = section.bed_elevation
        
        # 牛顿迭代
        h = 1.0  # 初始猜测
        
        for _ in range(20):
            A = section.area(h)
            if A <= 1e-6:
                A = 1e-6
            
            V = discharge / A
            H = z + h + V**2 / (2 * 9.81)
            
            # 残差
            residual = H - target_energy
            
            if abs(residual) < self.convergence_tol:
                break
            
            # 导数 dH/dh = 1 - Q²/(gA³) * dA/dh
            # 对于梯形断面： dA/dh = b + 2*m*h
            b = section.bottom_width
            m = section.side_slope
            dA_dh = b + 2 * m * h
            
            dH_dh = 1.0 - discharge**2 / (9.81 * A**3) * dA_dh
            
            if abs(dH_dh) < 1e-6:
                break
            
            # 牛顿更新
            h = h - residual / dH_dh
            h = max(h, self.min_depth)
            h = min(h, section.max_depth)
        
        return h
    
    def solve_normal_depth(self, section, discharge: float) -> float:
        """
        计算正常水深（均匀流）
        
        Parameters
        ----------
        section : ChannelSection
            断面
        discharge : float
            流量 (m³/s)
        
        Returns
        -------
        normal_depth : float
            正常水深 (m)
        """
        # 使用二分法求解 Manning 方程
        # Q = (A/n) * R^(2/3) * S0^(1/2)
        
        # 获取渠底坡度（从相邻断面估算）
        sections = self.channel.sections
        idx = sections.index(section)
        
        if idx < len(sections) - 1:
            dx = sections[idx + 1].station - section.station
            dz = section.bed_elevation - sections[idx + 1].bed_elevation
            S0 = dz / dx if dx > 0 else 0.0001
        else:
            S0 = 0.0001
        
        S0 = max(abs(S0), 1e-6)
        
        # 二分法
        h_min = self.min_depth
        h_max = section.max_depth
        
        for _ in range(50):
            h = (h_min + h_max) / 2.0
            
            A = section.area(h)
            R = section.hydraulic_radius(h)
            
            if R <= 0:
                h_min = h
                continue
            
            Q_calc = (A / section.manning_n) * (R ** (2.0 / 3.0)) * (S0 ** 0.5)
            
            if abs(Q_calc - discharge) < 0.01:
                return h
            
            if Q_calc < discharge:
                h_min = h
            else:
                h_max = h
        
        return (h_min + h_max) / 2.0

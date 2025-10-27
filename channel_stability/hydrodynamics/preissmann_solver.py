"""
Preissmann隐式格式水动力学求解器

基于圣维南方程组的一维非恒定流数值求解
集成自适应网格功能
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from ..core.channel_system import ChannelSystem
from .boundary_conditions import BoundaryConditions
from .adaptive_mesh import AdaptiveMeshRefiner, MeshQualityMetrics


@dataclass
class HydrodynamicResults:
    """水动力学计算结果"""
    times: np.ndarray  # 时间序列
    stations: np.ndarray  # 断面桩号
    depths: np.ndarray  # 水深 (时间×断面)
    discharges: np.ndarray  # 流量 (时间×断面)
    velocities: np.ndarray  # 流速 (时间×断面)
    water_levels: np.ndarray  # 水位 (时间×断面)
    froude_numbers: np.ndarray  # Froude数 (时间×断面)
    
    # 网格质量指标历史
    mesh_quality_history: list[MeshQualityMetrics] = field(default_factory=list)
    
    def get_time_slice(self, time_index: int) -> Dict[str, np.ndarray]:
        """获取某一时刻的数据"""
        return {
            'time': self.times[time_index],
            'stations': self.stations,
            'depths': self.depths[time_index],
            'discharges': self.discharges[time_index],
            'velocities': self.velocities[time_index],
            'water_levels': self.water_levels[time_index],
            'froude_numbers': self.froude_numbers[time_index],
        }
    
    def get_station_timeseries(self, station_index: int) -> Dict[str, np.ndarray]:
        """获取某一断面的时间序列数据"""
        return {
            'times': self.times,
            'station': self.stations[station_index],
            'depths': self.depths[:, station_index],
            'discharges': self.discharges[:, station_index],
            'velocities': self.velocities[:, station_index],
            'water_levels': self.water_levels[:, station_index],
            'froude_numbers': self.froude_numbers[:, station_index],
        }


class PreissmannHydrodynamicSolver:
    """Preissmann隐式格式水动力学求解器"""
    
    def __init__(
        self,
        channel: ChannelSystem,
        boundary_conditions: BoundaryConditions,
        theta: float = 0.7,
        max_iterations: int = 25,
        convergence_tol: float = 1e-4,
        min_depth: float = 1e-3,
        relaxation: float = 0.05,
        enable_adaptive_mesh: bool = True,
        adaptive_mesh_interval: int = 10,
    ):
        """
        初始化Preissmann求解器
        
        Parameters
        ----------
        channel : ChannelSystem
            明渠系统
        boundary_conditions : BoundaryConditions
            边界条件
        theta : float
            时间权重系数 (0.5=Crank-Nicolson, 1.0=全隐式)
        max_iterations : int
            Picard迭代最大次数
        convergence_tol : float
            收敛容差
        min_depth : float
            最小水深
        relaxation : float
            松弛因子
        enable_adaptive_mesh : bool
            是否启用自适应网格
        adaptive_mesh_interval : int
            自适应网格更新间隔（时间步）
        """
        self.channel = channel
        self.bc = boundary_conditions
        self.theta = theta
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.min_depth = min_depth
        self.relaxation = relaxation
        
        self.enable_adaptive_mesh = enable_adaptive_mesh
        self.adaptive_mesh_interval = adaptive_mesh_interval
        
        if self.enable_adaptive_mesh:
            self.mesh_refiner = AdaptiveMeshRefiner()
        else:
            self.mesh_refiner = None
    
    def _manning_discharge(self, section, depth: float, slope: float) -> float:
        """Manning公式计算流量"""
        if depth <= self.min_depth or slope <= 0:
            return 0.0
        
        area = section.area(depth)
        R = section.hydraulic_radius(depth)
        
        if R <= 0:
            return 0.0
        
        Q = (area / section.manning_n) * (R ** (2.0 / 3.0)) * (slope ** 0.5)
        return Q
    
    def _manning_friction_slope(self, section, depth: float, discharge: float) -> float:
        """Manning公式计算摩阻坡度"""
        if depth <= self.min_depth:
            return 0.0
        
        area = section.area(depth)
        if area <= 1e-6:
            return 0.0
        
        R = section.hydraulic_radius(depth)
        if R <= 0:
            return 0.0
        
        n = section.manning_n
        Sf = (n * discharge / area) ** 2 / (R ** (4.0 / 3.0))
        return Sf
    
    def solve(
        self,
        total_time: float,
        dt: float,
        initial_depth: float = 1.0,
        initial_discharge: float = 0.0,
        save_interval: Optional[int] = None,
    ) -> HydrodynamicResults:
        """
        求解非恒定流
        
        Parameters
        ----------
        total_time : float
            总模拟时间 (s)
        dt : float
            时间步长 (s)
        initial_depth : float
            初始水深 (m)
        initial_discharge : float
            初始流量 (m³/s)
        save_interval : int, optional
            保存结果的时间步间隔
        
        Returns
        -------
        results : HydrodynamicResults
            水动力学计算结果
        """
        if save_interval is None:
            save_interval = max(1, int(total_time / dt / 1000))  # 最多保存1000个时间步
        
        # 初始化断面
        stations = self.channel.stations
        sections = self.channel.sections
        num_sections = len(sections)
        
        bed_elevations = np.array([s.bed_elevation for s in sections])
        
        # 初始化状态变量
        depths = np.full(num_sections, initial_depth, dtype=float)
        discharges = np.full(num_sections, initial_discharge, dtype=float)
        
        # 时间步进
        num_steps = int(total_time / dt) + 1
        times = np.arange(0, total_time + dt, dt)
        
        # 存储结果（按save_interval采样）
        saved_times = []
        saved_depths = []
        saved_discharges = []
        mesh_quality_history = []
        
        # 保存初始状态
        saved_times.append(0.0)
        saved_depths.append(depths.copy())
        saved_discharges.append(discharges.copy())
        
        # 主时间循环
        for step in range(1, len(times)):
            t_curr = times[step - 1]
            t_next = times[step]
            
            old_depths = depths.copy()
            old_discharges = discharges.copy()
            
            # 应用边界条件
            q_up = max(self.bc.get_upstream_discharge(t_next), 0.0)
            h_up = self.bc.get_upstream_stage(t_next, q_up)
            
            discharges[0] = q_up
            depths[0] = max(h_up - bed_elevations[0], self.min_depth)
            
            # Picard迭代
            for iteration in range(self.max_iterations):
                prev_depths = depths.copy()
                prev_discharges = discharges.copy()
                
                # 更新内部节点
                for i in range(1, num_sections - 1):
                    dx = stations[i] - stations[i - 1]
                    section = sections[i]
                    
                    # 时间加权平均
                    depth_avg = self.theta * prev_depths[i] + (1 - self.theta) * old_depths[i]
                    depth_avg = max(depth_avg, self.min_depth)
                    
                    area_avg = section.area(depth_avg)
                    R = section.hydraulic_radius(depth_avg)
                    if R <= 0:
                        R = self.min_depth
                    
                    # 水面坡度
                    h_i = bed_elevations[i] + prev_depths[i]
                    h_im1 = bed_elevations[i - 1] + prev_depths[i - 1]
                    h_i_old = bed_elevations[i] + old_depths[i]
                    h_im1_old = bed_elevations[i - 1] + old_depths[i - 1]
                    
                    dHdx = (self.theta * (h_i - h_im1) + (1 - self.theta) * (h_i_old - h_im1_old)) / dx
                    
                    # 摩阻坡度
                    q_avg = self.theta * prev_discharges[i] + (1 - self.theta) * old_discharges[i]
                    Sf = self._manning_friction_slope(section, depth_avg, q_avg)
                    Sf = np.clip(Sf, 0.0, 5.0)
                    
                    # 动量方程
                    sign_q = np.sign(q_avg) if q_avg != 0 else 1.0
                    momentum_term = np.clip(dHdx + sign_q * Sf, -0.02, 0.02)
                    
                    target_q = old_discharges[i] - dt * 9.81 * area_avg * momentum_term
                    new_q = (1 - self.relaxation) * prev_discharges[i] + self.relaxation * target_q
                    discharges[i] = np.clip(new_q, -500.0, 500.0)
                
                # 连续性方程更新水深
                for i in range(1, num_sections - 1):
                    dx = stations[i] - stations[i - 1]
                    section = sections[i]
                    
                    area_old = section.area(old_depths[i])
                    flux = self.theta * (discharges[i] - discharges[i - 1]) + \
                           (1 - self.theta) * (old_discharges[i] - old_discharges[i - 1])
                    
                    area_new = area_old - (dt / dx) * flux
                    area_new = max(area_new, self.min_depth * section.bottom_width)
                    
                    depth_new = area_new / section.bottom_width if section.bottom_width > 0 else self.min_depth
                    depth_new = max(depth_new, self.min_depth)
                    depth_new = np.clip(depth_new, prev_depths[i] - 0.5, prev_depths[i] + 0.5)
                    
                    depths[i] = depth_new
                
                # 下游边界条件
                q_down = discharges[-2]
                h_down = self.bc.get_downstream_stage(t_next, q_down)
                depths[-1] = max(h_down - bed_elevations[-1], self.min_depth)
                discharges[-1] = discharges[-2]
                
                # 检查收敛性
                change_depth = np.max(np.abs(depths - prev_depths))
                change_q = np.max(np.abs(discharges - prev_discharges))
                
                if max(change_depth, change_q) < self.convergence_tol:
                    break
            
            # 自适应网格（每隔一定步数执行一次）
            if self.enable_adaptive_mesh and step % self.adaptive_mesh_interval == 0:
                areas = np.array([s.area(d) for s, d in zip(sections, depths)])
                top_widths = np.array([s.top_width(d) for s, d in zip(sections, depths)])
                
                metrics = self.mesh_refiner.evaluate_mesh_quality(
                    stations, depths, discharges, areas, top_widths
                )
                mesh_quality_history.append(metrics)
            
            # 保存结果
            if step % save_interval == 0:
                saved_times.append(t_next)
                saved_depths.append(depths.copy())
                saved_discharges.append(discharges.copy())
        
        # 构造结果
        saved_times = np.array(saved_times)
        saved_depths = np.array(saved_depths)
        saved_discharges = np.array(saved_discharges)
        
        # 计算派生量
        num_saved = len(saved_times)
        velocities = np.zeros_like(saved_depths)
        water_levels = np.zeros_like(saved_depths)
        froude_numbers = np.zeros_like(saved_depths)
        
        for i, section in enumerate(sections):
            for t in range(num_saved):
                depth = saved_depths[t, i]
                discharge = saved_discharges[t, i]
                
                area = section.area(depth)
                velocity = discharge / area if area > 1e-6 else 0.0
                velocities[t, i] = velocity
                
                water_levels[t, i] = section.bed_elevation + depth
                
                top_width = section.top_width(depth)
                if depth > 1e-3 and top_width > 1e-3:
                    froude_numbers[t, i] = velocity / np.sqrt(9.81 * area / top_width)
                else:
                    froude_numbers[t, i] = 0.0
        
        return HydrodynamicResults(
            times=saved_times,
            stations=stations,
            depths=saved_depths,
            discharges=saved_discharges,
            velocities=velocities,
            water_levels=water_levels,
            froude_numbers=froude_numbers,
            mesh_quality_history=mesh_quality_history,
        )

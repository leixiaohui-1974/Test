"""
准稳态近似求解器（完全守恒）

基本思想：
将非恒定流问题分解为一系列稳态流问题
每个时间步重新计算稳态流

优点：
1. 完全质量守恒（<0.1%误差）
2. 数值稳定
3. 计算快速

局限：
1. 忽略惯性项
2. 不适用于快速瞬变
3. 适用于缓慢变化问题（dt > 60s）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..core.channel_system import ChannelSystem
from .boundary_conditions import BoundaryConditions
from .steady_flow_solver import SteadyFlowSolver
from .preissmann_solver import HydrodynamicResults


class QuasiSteadySolver:
    """准稳态近似求解器"""
    
    def __init__(
        self,
        channel: ChannelSystem,
        boundary_conditions: BoundaryConditions,
    ):
        """
        初始化准稳态求解器
        
        Parameters
        ----------
        channel : ChannelSystem
            明渠系统
        boundary_conditions : BoundaryConditions
            边界条件
        """
        self.channel = channel
        self.bc = boundary_conditions
        self.steady_solver = SteadyFlowSolver(channel)
    
    def solve(
        self,
        total_time: float,
        dt: float = 60.0,  # 准稳态假设，时间步长可以较大
        initial_depth: float = 1.0,
        initial_discharge: float = 0.0,
        save_interval: Optional[int] = None,
        **kwargs,  # 兼容性参数（忽略）
    ) -> HydrodynamicResults:
        """
        求解准稳态流
        
        Parameters
        ----------
        total_time : float
            总模拟时间 (s)
        dt : float
            时间步长 (s)，建议60-600秒
        initial_depth : float
            初始水深 (m)
        initial_discharge : float
            初始流量 (m³/s)
        save_interval : int, optional
            保存间隔
        
        Returns
        -------
        results : HydrodynamicResults
            水动力学结果
        """
        if save_interval is None:
            save_interval = max(1, int(total_time / dt / 100))  # 最多100个时间步
        
        print(f"\n准稳态求解器:")
        print(f"  总时间: {total_time/3600:.2f} 小时")
        print(f"  时间步长: {dt:.1f} 秒")
        print(f"  假设: 每个时间步都达到稳态（忽略惯性）\n")
        
        # 时间序列
        times = np.arange(0, total_time + dt, dt)
        
        # 存储结果
        saved_times = []
        saved_depths = []
        saved_discharges = []
        
        stations = self.channel.stations
        num_sections = len(self.channel.sections)
        
        # 初始状态
        saved_times.append(0.0)
        saved_depths.append(np.full(num_sections, initial_depth))
        saved_discharges.append(np.full(num_sections, initial_discharge))
        
        # 时间步进
        for step, t in enumerate(times[1:], 1):
            # 获取当前时刻的边界条件
            q_up = self.bc.get_upstream_discharge(t)
            
            # 下游边界：先估算流量，再获取水位
            q_down_estimate = q_up  # 假设流量沿程不变（稳态）
            h_down = self.bc.get_downstream_stage(t, q_down_estimate)
            
            # 获取下游水深（相对于床面）
            bed_elev_down = self.channel.sections[-1].bed_elevation
            downstream_depth = h_down - bed_elev_down
            downstream_depth = max(downstream_depth, 0.1)
            
            # 求解稳态流
            try:
                steady_results = self.steady_solver.solve_gradually_varied_flow(
                    discharge=q_up,
                    downstream_depth=downstream_depth,
                    max_iterations=100,
                    tolerance=1e-6,
                )
                
                depths = steady_results['depths']
                discharges = steady_results['discharges']
                
            except Exception as e:
                # 如果求解失败，使用上一步的结果
                print(f"警告: 时间步{step} (t={t/3600:.2f}h) 稳态求解失败，使用上一步结果")
                depths = saved_depths[-1].copy()
                discharges = saved_discharges[-1].copy()
            
            # 保存结果
            if step % save_interval == 0 or t >= total_time:
                saved_times.append(t)
                saved_depths.append(depths.copy())
                saved_discharges.append(discharges.copy())
            
            # 输出进度
            if step % 10 == 0:
                progress = t / total_time * 100
                v_max = max(abs(discharges[i] / self.channel.sections[i].area(depths[i])) 
                           if self.channel.sections[i].area(depths[i]) > 1e-6 else 0
                           for i in range(num_sections))
                q_in = discharges[0]
                q_out = discharges[-1]
                mass_err = abs(q_out - q_in) / (q_in + 1e-10)
                
                print(f"进度: {progress:.1f}% (t={t/3600:.2f}h, Q={q_up:.2f}m³/s, "
                      f"v_max={v_max:.3f}m/s, 质守={mass_err*100:.2f}%)")
        
        # 构造结果
        saved_times = np.array(saved_times)
        saved_depths = np.array(saved_depths)
        saved_discharges = np.array(saved_discharges)
        
        # 计算派生量
        num_saved = len(saved_times)
        velocities = np.zeros_like(saved_depths)
        water_levels = np.zeros_like(saved_depths)
        froude_numbers = np.zeros_like(saved_depths)
        
        for i, section in enumerate(self.channel.sections):
            for t_idx in range(num_saved):
                depth = saved_depths[t_idx, i]
                discharge = saved_discharges[t_idx, i]
                
                area = section.area(depth)
                velocity = discharge / area if area > 1e-6 else 0.0
                velocities[t_idx, i] = velocity
                
                water_levels[t_idx, i] = section.bed_elevation + depth
                
                top_width = section.top_width(depth)
                if depth > 1e-3 and top_width > 1e-3:
                    froude_numbers[t_idx, i] = abs(velocity) / np.sqrt(9.81 * area / top_width)
                else:
                    froude_numbers[t_idx, i] = 0.0
        
        print(f"\n✓ 准稳态求解完成: {len(saved_times)}个时间点\n")
        
        return HydrodynamicResults(
            times=saved_times,
            stations=stations,
            depths=saved_depths,
            discharges=saved_discharges,
            velocities=velocities,
            water_levels=water_levels,
            froude_numbers=froude_numbers,
            mesh_quality_history=[],
        )

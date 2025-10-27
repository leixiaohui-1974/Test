"""
智能边界条件管理器

自动处理阶跃变化，确保数值稳定性和质量守恒
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from .boundary_conditions import BoundaryConditions


class SmartBoundaryConditions:
    """
    智能边界条件管理器
    
    自动特性：
    1. 阶跃变化自动平滑过渡
    2. 自适应过渡时间
    3. 确保数值稳定性
    """
    
    def __init__(
        self,
        base_bc: BoundaryConditions,
        auto_smooth: bool = True,
        min_transition_time: float = 300.0,  # 最小过渡时间5分钟
        max_change_rate: float = 0.5,  # 最大变化率（50%/小时）
    ):
        """
        初始化智能边界条件
        
        Parameters
        ----------
        base_bc : BoundaryConditions
            基础边界条件
        auto_smooth : bool
            是否自动平滑阶跃变化
        min_transition_time : float
            最小过渡时间（秒）
        max_change_rate : float
            最大允许变化率（相对值/小时）
        """
        self.base_bc = base_bc
        self.auto_smooth = auto_smooth
        self.min_transition_time = min_transition_time
        self.max_change_rate = max_change_rate
        
        # 缓存历史值，用于检测阶跃
        self.prev_time = None
        self.prev_q_up = None
        self.prev_h_down = None
        
        # 平滑过渡状态
        self.transition_active = False
        self.transition_start_time = None
        self.transition_duration = None
        self.value_before = None
        self.value_after = None
        self.transition_type = None  # 'discharge' or 'stage'
    
    def _detect_step_change(
        self,
        time: float,
        current_value: float,
        prev_value: Optional[float],
    ) -> tuple[bool, float]:
        """
        检测阶跃变化
        
        Returns
        -------
        is_step : bool
            是否检测到阶跃
        change_magnitude : float
            变化幅度（相对值）
        """
        if prev_value is None:
            return False, 0.0
        
        change = abs(current_value - prev_value)
        relative_change = change / (abs(prev_value) + 1e-10)
        
        # 如果变化超过5%且时间间隔很短（<60秒），认为是阶跃
        if self.prev_time is not None:
            dt = time - self.prev_time
            if dt < 60.0 and relative_change > 0.05:
                return True, relative_change
        
        return False, relative_change
    
    def _compute_transition_duration(self, change_magnitude: float, base_value: float) -> float:
        """
        计算合适的过渡时间
        
        Parameters
        ----------
        change_magnitude : float
            变化幅度（相对值）
        base_value : float
            基准值
        
        Returns
        -------
        duration : float
            过渡时间（秒）
        """
        # 根据变化幅度自适应过渡时间
        # 变化越大，过渡时间越长
        
        if change_magnitude < 0.2:  # <20%变化
            duration = self.min_transition_time
        elif change_magnitude < 0.5:  # 20-50%变化
            duration = 2 * self.min_transition_time
        elif change_magnitude < 1.0:  # 50-100%变化
            duration = 3 * self.min_transition_time
        else:  # >100%变化
            duration = 5 * self.min_transition_time
        
        # 确保不超过最大变化率限制
        # max_change_rate是每小时的相对变化
        # duration = change_magnitude / (max_change_rate / 3600)
        min_duration_by_rate = change_magnitude * 3600 / self.max_change_rate
        duration = max(duration, min_duration_by_rate)
        
        return duration
    
    def _smooth_transition(
        self,
        t: float,
        t_start: float,
        duration: float,
        value_before: float,
        value_after: float,
    ) -> float:
        """
        S曲线平滑过渡
        
        使用三次多项式：3x² - 2x³
        """
        if t < t_start:
            return value_before
        elif t >= t_start + duration:
            return value_after
        else:
            progress = (t - t_start) / duration
            # S曲线（光滑，导数连续）
            smooth_factor = 3 * progress**2 - 2 * progress**3
            return value_before + (value_after - value_before) * smooth_factor
    
    def get_upstream_discharge(self, time: float) -> float:
        """
        获取上游流量（带自动平滑）
        
        Parameters
        ----------
        time : float
            当前时间
        
        Returns
        -------
        discharge : float
            上游流量
        """
        # 从基础边界条件获取原始值
        raw_value = self.base_bc.get_upstream_discharge(time)
        
        if not self.auto_smooth:
            return raw_value
        
        # 检测阶跃
        is_step, change_mag = self._detect_step_change(time, raw_value, self.prev_q_up)
        
        if is_step and not self.transition_active:
            # 开始新的过渡
            self.transition_active = True
            self.transition_start_time = time
            self.value_before = self.prev_q_up
            self.value_after = raw_value
            self.transition_duration = self._compute_transition_duration(change_mag, self.prev_q_up)
            self.transition_type = 'discharge'
            
            print(f"\n检测到流量阶跃: {self.value_before:.2f}→{self.value_after:.2f} m³/s "
                  f"({change_mag*100:.1f}%变化)")
            print(f"  自动平滑过渡时间: {self.transition_duration/60:.1f}分钟")
        
        # 如果正在过渡中
        if self.transition_active and self.transition_type == 'discharge':
            if time >= self.transition_start_time + self.transition_duration:
                # 过渡完成
                self.transition_active = False
                smoothed_value = self.value_after
            else:
                # 应用平滑过渡
                smoothed_value = self._smooth_transition(
                    time, self.transition_start_time, self.transition_duration,
                    self.value_before, self.value_after
                )
            
            # 更新缓存
            self.prev_time = time
            self.prev_q_up = smoothed_value
            return smoothed_value
        else:
            # 正常返回
            self.prev_time = time
            self.prev_q_up = raw_value
            return raw_value
    
    def get_downstream_stage(self, time: float, discharge: float) -> float:
        """
        获取下游水位（带自动平滑）
        
        Parameters
        ----------
        time : float
            当前时间
        discharge : float
            流量
        
        Returns
        -------
        stage : float
            下游水位
        """
        raw_value = self.base_bc.get_downstream_stage(time, discharge)
        
        if not self.auto_smooth:
            return raw_value
        
        # 检测阶跃
        is_step, change_mag = self._detect_step_change(time, raw_value, self.prev_h_down)
        
        if is_step and not self.transition_active:
            # 开始新的过渡
            self.transition_active = True
            self.transition_start_time = time
            self.value_before = self.prev_h_down
            self.value_after = raw_value
            
            # 水位变化的过渡时间（相对于床面）
            # 假设床面高程100m，水位变化1m相当于1/100的相对变化
            relative_change = change_mag  # 已经是相对于床面的变化
            self.transition_duration = self._compute_transition_duration(relative_change, 1.0)
            self.transition_type = 'stage'
            
            print(f"\n检测到水位阶跃: {self.value_before:.2f}→{self.value_after:.2f} m "
                  f"({change_mag*100:.1f}%变化)")
            print(f"  自动平滑过渡时间: {self.transition_duration/60:.1f}分钟")
        
        # 如果正在过渡中
        if self.transition_active and self.transition_type == 'stage':
            if time >= self.transition_start_time + self.transition_duration:
                # 过渡完成
                self.transition_active = False
                smoothed_value = self.value_after
            else:
                # 应用平滑过渡
                smoothed_value = self._smooth_transition(
                    time, self.transition_start_time, self.transition_duration,
                    self.value_before, self.value_after
                )
            
            # 更新缓存
            self.prev_time = time
            self.prev_h_down = smoothed_value
            return smoothed_value
        else:
            # 正常返回
            self.prev_time = time
            self.prev_h_down = raw_value
            return raw_value
    
    def get_upstream_stage(self, time: float, discharge: float) -> float:
        """获取上游水位"""
        return self.base_bc.get_upstream_stage(time, discharge)


def create_smart_bc(
    upstream_q_func: Callable[[float], float],
    downstream_h_func: Callable[[float, float], float],
    auto_smooth: bool = True,
) -> SmartBoundaryConditions:
    """
    创建智能边界条件（快捷方式）
    
    Parameters
    ----------
    upstream_q_func : Callable
        上游流量函数 f(t) -> Q
    downstream_h_func : Callable
        下游水位函数 f(t, Q) -> h
    auto_smooth : bool
        是否自动平滑
    
    Returns
    -------
    smart_bc : SmartBoundaryConditions
        智能边界条件
    """
    base_bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=upstream_q_func,
        downstream_stage_func=downstream_h_func,
    )
    
    return SmartBoundaryConditions(
        base_bc=base_bc,
        auto_smooth=auto_smooth,
    )

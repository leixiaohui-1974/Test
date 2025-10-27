"""
边界条件处理

定义明渠上下游边界条件
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
import numpy as np


@dataclass
class BoundaryConditions:
    """边界条件"""
    
    # 上游边界条件
    upstream_type: str = "discharge"  # "discharge" 或 "stage"
    upstream_discharge_func: Optional[Callable[[float], float]] = None  # 流量过程
    upstream_stage_func: Optional[Callable[[float, float], float]] = None  # 水位-流量关系
    
    # 下游边界条件
    downstream_type: str = "stage"  # "stage", "rating_curve", 或 "free"
    downstream_stage_func: Optional[Callable[[float, float], float]] = None  # 水位过程或水位-流量关系
    
    # 常数边界条件
    upstream_constant_q: float = 10.0  # 恒定上游流量 (m³/s)
    upstream_constant_h: float = 2.0  # 恒定上游水位 (m)
    downstream_constant_h: float = 1.5  # 恒定下游水位 (m)
    
    def __post_init__(self):
        """初始化默认边界条件函数"""
        if self.upstream_discharge_func is None:
            self.upstream_discharge_func = lambda t: self.upstream_constant_q
        
        if self.upstream_stage_func is None:
            self.upstream_stage_func = lambda t, q: self.upstream_constant_h
        
        if self.downstream_stage_func is None:
            self.downstream_stage_func = lambda t, q: self.downstream_constant_h
    
    def get_upstream_discharge(self, time: float) -> float:
        """获取上游流量"""
        if self.upstream_type == "discharge" and self.upstream_discharge_func:
            return self.upstream_discharge_func(time)
        return self.upstream_constant_q
    
    def get_upstream_stage(self, time: float, discharge: float) -> float:
        """获取上游水位"""
        if self.upstream_stage_func:
            return self.upstream_stage_func(time, discharge)
        return self.upstream_constant_h
    
    def get_downstream_stage(self, time: float, discharge: float) -> float:
        """获取下游水位"""
        if self.downstream_stage_func:
            return self.downstream_stage_func(time, discharge)
        return self.downstream_constant_h
    
    @staticmethod
    def create_constant_bc(
        upstream_q: float = 10.0,
        downstream_h: float = 1.5,
    ) -> BoundaryConditions:
        """创建恒定边界条件"""
        return BoundaryConditions(
            upstream_type="discharge",
            downstream_type="stage",
            upstream_constant_q=upstream_q,
            downstream_constant_h=downstream_h,
        )
    
    @staticmethod
    def create_hydrograph_bc(
        time_series: List[Tuple[float, float]],
        downstream_h: float = 1.5,
    ) -> BoundaryConditions:
        """
        创建上游流量过程边界条件
        
        Parameters
        ----------
        time_series : List[Tuple[float, float]]
            时间-流量序列 [(t1, q1), (t2, q2), ...]
        downstream_h : float
            下游水位
        """
        if not time_series:
            raise ValueError("流量过程时间序列不能为空")
        
        times = np.array([t for t, q in time_series])
        discharges = np.array([q for t, q in time_series])
        
        def discharge_func(t: float) -> float:
            return float(np.interp(t, times, discharges))
        
        return BoundaryConditions(
            upstream_type="discharge",
            downstream_type="stage",
            upstream_discharge_func=discharge_func,
            downstream_constant_h=downstream_h,
        )
    
    @staticmethod
    def create_rating_curve_bc(
        upstream_q: float,
        downstream_rating: List[Tuple[float, float]],
    ) -> BoundaryConditions:
        """
        创建下游水位-流量关系边界条件
        
        Parameters
        ----------
        upstream_q : float
            上游流量
        downstream_rating : List[Tuple[float, float]]
            下游水位-流量关系 [(q1, h1), (q2, h2), ...]
        """
        if not downstream_rating:
            raise ValueError("水位-流量关系不能为空")
        
        qs = np.array([q for q, h in downstream_rating])
        hs = np.array([h for q, h in downstream_rating])
        
        def stage_func(t: float, q: float) -> float:
            return float(np.interp(q, qs, hs))
        
        return BoundaryConditions(
            upstream_type="discharge",
            downstream_type="rating_curve",
            upstream_constant_q=upstream_q,
            downstream_stage_func=stage_func,
        )

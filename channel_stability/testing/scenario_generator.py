"""
阶跃变化场景生成器

生成各种阶跃变化边界条件，用于测试系统的动态响应特性
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class StepChangeScenario:
    """阶跃变化场景"""
    name: str
    description: str
    parameter_type: str  # 'discharge', 'stage', 'water_quality', 'rainfall', 'groundwater'
    time_function: Callable[[float], float]
    baseline_value: float
    step_value: float
    step_time: float
    
    def get_value(self, time: float) -> float:
        """获取指定时刻的参数值"""
        return self.time_function(time)


class ScenarioGenerator:
    """场景生成器"""
    
    @staticmethod
    def create_discharge_step(
        baseline_q: float = 10.0,
        step_q: float = 20.0,
        step_time: float = 3600.0,
        name: str = "流量阶跃变化"
    ) -> StepChangeScenario:
        """
        创建流量阶跃变化场景
        
        Parameters
        ----------
        baseline_q : float
            基准流量 (m³/s)
        step_q : float
            阶跃后流量 (m³/s)
        step_time : float
            阶跃时刻 (s)
        name : str
            场景名称
        """
        def q_func(t):
            return step_q if t >= step_time else baseline_q
        
        return StepChangeScenario(
            name=name,
            description=f"流量从 {baseline_q:.1f} m³/s 阶跃至 {step_q:.1f} m³/s",
            parameter_type='discharge',
            time_function=q_func,
            baseline_value=baseline_q,
            step_value=step_q,
            step_time=step_time,
        )
    
    @staticmethod
    def create_stage_step(
        baseline_h: float = 2.0,
        step_h: float = 3.0,
        step_time: float = 3600.0,
        name: str = "水位阶跃变化"
    ) -> StepChangeScenario:
        """
        创建水位阶跃变化场景
        
        Parameters
        ----------
        baseline_h : float
            基准水深 (m)
        step_h : float
            阶跃后水深 (m)
        step_time : float
            阶跃时刻 (s)
        name : str
            场景名称
        """
        def h_func(t):
            return step_h if t >= step_time else baseline_h
        
        return StepChangeScenario(
            name=name,
            description=f"水深从 {baseline_h:.1f} m 阶跃至 {step_h:.1f} m",
            parameter_type='stage',
            time_function=h_func,
            baseline_value=baseline_h,
            step_value=step_h,
            step_time=step_time,
        )
    
    @staticmethod
    def create_water_quality_step(
        indicator: str = 'BOD',
        baseline_conc: float = 5.0,
        step_conc: float = 15.0,
        step_time: float = 3600.0,
        name: str = "水质阶跃变化"
    ) -> StepChangeScenario:
        """
        创建水质指标阶跃变化场景
        
        Parameters
        ----------
        indicator : str
            水质指标名称
        baseline_conc : float
            基准浓度 (mg/L)
        step_conc : float
            阶跃后浓度 (mg/L)
        step_time : float
            阶跃时刻 (s)
        name : str
            场景名称
        """
        def conc_func(t):
            return step_conc if t >= step_time else baseline_conc
        
        return StepChangeScenario(
            name=f"{name}-{indicator}",
            description=f"{indicator}浓度从 {baseline_conc:.1f} mg/L 阶跃至 {step_conc:.1f} mg/L",
            parameter_type='water_quality',
            time_function=conc_func,
            baseline_value=baseline_conc,
            step_value=step_conc,
            step_time=step_time,
        )
    
    @staticmethod
    def create_rainfall_step(
        baseline_rainfall: float = 0.0,
        step_rainfall: float = 10.0,
        step_time: float = 3600.0,
        duration: float = 1800.0,
        name: str = "降雨阶跃变化"
    ) -> StepChangeScenario:
        """
        创建降雨阶跃变化场景
        
        Parameters
        ----------
        baseline_rainfall : float
            基准降雨强度 (mm/h)
        step_rainfall : float
            降雨强度 (mm/h)
        step_time : float
            降雨开始时刻 (s)
        duration : float
            降雨持续时间 (s)
        name : str
            场景名称
        """
        def rainfall_func(t):
            if step_time <= t < step_time + duration:
                return step_rainfall
            else:
                return baseline_rainfall
        
        return StepChangeScenario(
            name=name,
            description=f"降雨强度从 {baseline_rainfall:.1f} mm/h 阶跃至 {step_rainfall:.1f} mm/h，持续 {duration/60:.0f} 分钟",
            parameter_type='rainfall',
            time_function=rainfall_func,
            baseline_value=baseline_rainfall,
            step_value=step_rainfall,
            step_time=step_time,
        )
    
    @staticmethod
    def create_groundwater_step(
        baseline_level: float = 98.0,
        step_level: float = 99.5,
        step_time: float = 3600.0,
        name: str = "地下水位阶跃变化"
    ) -> StepChangeScenario:
        """
        创建地下水位阶跃变化场景
        
        Parameters
        ----------
        baseline_level : float
            基准地下水位 (m)
        step_level : float
            阶跃后地下水位 (m)
        step_time : float
            阶跃时刻 (s)
        name : str
            场景名称
        """
        def level_func(t):
            return step_level if t >= step_time else baseline_level
        
        return StepChangeScenario(
            name=name,
            description=f"地下水位从 {baseline_level:.1f} m 阶跃至 {step_level:.1f} m",
            parameter_type='groundwater',
            time_function=level_func,
            baseline_value=baseline_level,
            step_value=step_level,
            step_time=step_time,
        )
    
    @staticmethod
    def create_multiple_steps(
        parameter_type: str,
        baseline: float,
        step_values: List[float],
        step_times: List[float],
        name: str = "多次阶跃变化"
    ) -> StepChangeScenario:
        """
        创建多次阶跃变化场景
        
        Parameters
        ----------
        parameter_type : str
            参数类型
        baseline : float
            基准值
        step_values : List[float]
            各次阶跃的值
        step_times : List[float]
            各次阶跃的时刻
        name : str
            场景名称
        """
        def multi_step_func(t):
            value = baseline
            for step_t, step_v in zip(step_times, step_values):
                if t >= step_t:
                    value = step_v
            return value
        
        return StepChangeScenario(
            name=name,
            description=f"{parameter_type}多次阶跃变化：{len(step_values)}次",
            parameter_type=parameter_type,
            time_function=multi_step_func,
            baseline_value=baseline,
            step_value=step_values[-1] if step_values else baseline,
            step_time=step_times[0] if step_times else 0.0,
        )
    
    @staticmethod
    def create_ramp_change(
        parameter_type: str,
        baseline: float,
        final_value: float,
        ramp_start: float,
        ramp_duration: float,
        name: str = "斜坡变化"
    ) -> StepChangeScenario:
        """
        创建斜坡变化场景（线性渐变）
        
        Parameters
        ----------
        parameter_type : str
            参数类型
        baseline : float
            基准值
        final_value : float
            最终值
        ramp_start : float
            斜坡开始时刻 (s)
        ramp_duration : float
            斜坡持续时间 (s)
        name : str
            场景名称
        """
        def ramp_func(t):
            if t < ramp_start:
                return baseline
            elif t < ramp_start + ramp_duration:
                progress = (t - ramp_start) / ramp_duration
                return baseline + (final_value - baseline) * progress
            else:
                return final_value
        
        return StepChangeScenario(
            name=name,
            description=f"{parameter_type}从 {baseline:.1f} 线性变化至 {final_value:.1f}，历时 {ramp_duration/60:.0f} 分钟",
            parameter_type=parameter_type,
            time_function=ramp_func,
            baseline_value=baseline,
            step_value=final_value,
            step_time=ramp_start,
        )
    
    @staticmethod
    def create_combined_scenarios() -> Dict[str, List[StepChangeScenario]]:
        """
        创建组合测试场景集合
        
        Returns
        -------
        scenarios : Dict[str, List[StepChangeScenario]]
            场景集合，按类型分组
        """
        scenarios = {
            'discharge': [
                ScenarioGenerator.create_discharge_step(
                    baseline_q=10.0, step_q=20.0, step_time=1800.0,
                    name="流量增加阶跃"
                ),
                ScenarioGenerator.create_discharge_step(
                    baseline_q=20.0, step_q=10.0, step_time=1800.0,
                    name="流量减少阶跃"
                ),
                ScenarioGenerator.create_ramp_change(
                    parameter_type='discharge',
                    baseline=10.0, final_value=25.0,
                    ramp_start=1800.0, ramp_duration=1800.0,
                    name="流量斜坡增加"
                ),
            ],
            'stage': [
                ScenarioGenerator.create_stage_step(
                    baseline_h=2.0, step_h=3.0, step_time=1800.0,
                    name="水位上升阶跃"
                ),
                ScenarioGenerator.create_stage_step(
                    baseline_h=3.0, step_h=2.0, step_time=1800.0,
                    name="水位下降阶跃"
                ),
            ],
            'water_quality': [
                ScenarioGenerator.create_water_quality_step(
                    indicator='BOD',
                    baseline_conc=5.0, step_conc=15.0, step_time=1800.0,
                    name="BOD浓度阶跃"
                ),
                ScenarioGenerator.create_water_quality_step(
                    indicator='NH3N',
                    baseline_conc=1.0, step_conc=5.0, step_time=1800.0,
                    name="氨氮浓度阶跃"
                ),
            ],
            'rainfall': [
                ScenarioGenerator.create_rainfall_step(
                    baseline_rainfall=0.0, step_rainfall=20.0,
                    step_time=1800.0, duration=1800.0,
                    name="突发强降雨"
                ),
            ],
            'groundwater': [
                ScenarioGenerator.create_groundwater_step(
                    baseline_level=98.0, step_level=99.5,
                    step_time=1800.0,
                    name="地下水位上升"
                ),
                ScenarioGenerator.create_groundwater_step(
                    baseline_level=99.0, step_level=97.5,
                    step_time=1800.0,
                    name="地下水位下降"
                ),
            ],
        }
        
        return scenarios

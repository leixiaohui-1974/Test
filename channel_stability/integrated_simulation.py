"""
综合仿真系统

集成水动力学、水质和边坡稳定性的完整仿真流程
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .core.channel_system import ChannelSystem
from .core.monitoring_network import MonitoringNetwork
from .hydrodynamics.preissmann_solver import (
    PreissmannHydrodynamicSolver,
    HydrodynamicResults,
)
from .hydrodynamics.boundary_conditions import BoundaryConditions
from .water_quality.advection_diffusion_solver import (
    WaterQualitySolver,
    WaterQualityResults,
)
from .water_quality.reaction_kinetics import WaterQualityParameters
from .slope_stability.stability_calculator import (
    SlopeStabilityCalculator,
    StabilityResults,
)


@dataclass
class SimulationConfig:
    """仿真配置"""
    total_time: float = 86400.0  # 总时间 (s)，默认1天
    dt: float = 60.0  # 时间步长 (s)
    save_interval: int = 10  # 保存间隔
    
    # 水动力学参数
    enable_hydrodynamics: bool = True
    initial_depth: float = 2.0  # 初始水深 (m)
    initial_discharge: float = 10.0  # 初始流量 (m³/s)
    
    # 水质参数
    enable_water_quality: bool = True
    simulated_indicators: List[str] = field(default_factory=lambda: ['DO', 'BOD', 'NH3N', 'TN', 'TP'])
    
    # 边坡稳定性参数
    enable_slope_stability: bool = True
    stability_time_interval: int = 100  # 稳定性计算时间间隔
    
    # 降阶模型
    enable_reduced_order: bool = False
    reduced_order_type: str = "linear"  # "linear" or "deep_learning"


@dataclass
class IntegratedResults:
    """集成仿真结果"""
    hydrodynamics: Optional[HydrodynamicResults] = None
    water_quality: Optional[WaterQualityResults] = None
    slope_stability: Optional[StabilityResults] = None
    
    def summary(self) -> Dict[str, any]:
        """生成结果摘要"""
        summary = {}
        
        if self.hydrodynamics is not None:
            summary['hydrodynamics'] = {
                'num_times': len(self.hydrodynamics.times),
                'num_stations': len(self.hydrodynamics.stations),
                'max_depth': float(np.max(self.hydrodynamics.depths)),
                'max_velocity': float(np.max(self.hydrodynamics.velocities)),
                'max_froude': float(np.max(self.hydrodynamics.froude_numbers)),
            }
        
        if self.water_quality is not None:
            summary['water_quality'] = {
                'indicators': list(self.water_quality.concentrations.keys()),
                'num_times': len(self.water_quality.times),
                'num_stations': len(self.water_quality.stations),
            }
            for indicator, conc in self.water_quality.concentrations.items():
                summary['water_quality'][f'{indicator}_max'] = float(np.max(conc))
                summary['water_quality'][f'{indicator}_min'] = float(np.min(conc))
        
        if self.slope_stability is not None:
            summary['slope_stability'] = {
                'num_times': len(self.slope_stability.times),
                'num_stations': len(self.slope_stability.stations),
                'min_stability_factor': float(np.min(self.slope_stability.comprehensive_factors)),
                'max_unstable_sections': int(np.max(self.slope_stability.unstable_section_count)),
                'overall_stability_index': float(np.mean(self.slope_stability.channel_stability_index)),
            }
        
        return summary


class IntegratedSimulator:
    """综合仿真器"""
    
    def __init__(
        self,
        channel: ChannelSystem,
        monitoring_network: MonitoringNetwork,
        boundary_conditions: BoundaryConditions,
        water_quality_params: Optional[WaterQualityParameters] = None,
    ):
        """
        初始化综合仿真器
        
        Parameters
        ----------
        channel : ChannelSystem
            明渠系统
        monitoring_network : MonitoringNetwork
            监测网络
        boundary_conditions : BoundaryConditions
            边界条件
        water_quality_params : WaterQualityParameters, optional
            水质参数
        """
        self.channel = channel
        self.monitoring_network = monitoring_network
        self.boundary_conditions = boundary_conditions
        
        if water_quality_params is None:
            self.water_quality_params = WaterQualityParameters()
        else:
            self.water_quality_params = water_quality_params
        
        # 初始化各模块求解器
        self.hydro_solver = None
        self.wq_solver = None
        self.stability_calculator = None
    
    def run_simulation(
        self,
        config: Optional[SimulationConfig] = None,
    ) -> IntegratedResults:
        """
        运行综合仿真
        
        Parameters
        ----------
        config : SimulationConfig, optional
            仿真配置
        
        Returns
        -------
        results : IntegratedResults
            集成仿真结果
        """
        if config is None:
            config = SimulationConfig()
        
        results = IntegratedResults()
        
        # 1. 水动力学模拟
        if config.enable_hydrodynamics:
            print("正在运行水动力学模拟...")
            results.hydrodynamics = self._run_hydrodynamics(config)
            print(f"  完成，计算了 {len(results.hydrodynamics.times)} 个时间步")
        
        # 2. 水质模拟
        if config.enable_water_quality and results.hydrodynamics is not None:
            print("正在运行水质模拟...")
            results.water_quality = self._run_water_quality(
                config,
                results.hydrodynamics,
            )
            print(f"  完成，模拟了 {len(config.simulated_indicators)} 个指标")
        
        # 3. 边坡稳定性计算
        if config.enable_slope_stability and results.hydrodynamics is not None:
            print("正在计算边坡稳定性...")
            results.slope_stability = self._run_slope_stability(
                config,
                results.hydrodynamics,
            )
            print(f"  完成，计算了 {len(results.slope_stability.stations)} 个断面")
        
        return results
    
    def _run_hydrodynamics(
        self,
        config: SimulationConfig,
    ) -> HydrodynamicResults:
        """运行水动力学模拟"""
        self.hydro_solver = PreissmannHydrodynamicSolver(
            channel=self.channel,
            boundary_conditions=self.boundary_conditions,
            enable_adaptive_mesh=False,  # 简化，不启用自适应网格
        )
        
        results = self.hydro_solver.solve(
            total_time=config.total_time,
            dt=config.dt,
            initial_depth=config.initial_depth,
            initial_discharge=config.initial_discharge,
            save_interval=config.save_interval,
        )
        
        return results
    
    def _run_water_quality(
        self,
        config: SimulationConfig,
        hydro_results: HydrodynamicResults,
    ) -> WaterQualityResults:
        """运行水质模拟"""
        self.wq_solver = WaterQualitySolver(
            parameters=self.water_quality_params,
        )
        
        results = self.wq_solver.solve(
            hydrodynamic_results=hydro_results,
            indicators=config.simulated_indicators,
            dt=config.dt,
        )
        
        return results
    
    def _run_slope_stability(
        self,
        config: SimulationConfig,
        hydro_results: HydrodynamicResults,
    ) -> StabilityResults:
        """运行边坡稳定性计算"""
        self.stability_calculator = SlopeStabilityCalculator(
            channel=self.channel,
            monitoring_network=self.monitoring_network,
        )
        
        # 选择计算时间步
        num_times = len(hydro_results.times)
        time_indices = list(range(0, num_times, config.stability_time_interval))
        
        results = self.stability_calculator.compute_stability(
            hydrodynamic_results=hydro_results,
            time_indices=time_indices,
        )
        
        return results
    
    def export_results(
        self,
        results: IntegratedResults,
        output_dir: str,
    ) -> None:
        """
        导出结果
        
        Parameters
        ----------
        results : IntegratedResults
            仿真结果
        output_dir : str
            输出目录
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出摘要
        summary = results.summary()
        with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 导出水动力学结果
        if results.hydrodynamics is not None:
            np.savez(
                os.path.join(output_dir, 'hydrodynamics.npz'),
                times=results.hydrodynamics.times,
                stations=results.hydrodynamics.stations,
                depths=results.hydrodynamics.depths,
                discharges=results.hydrodynamics.discharges,
                velocities=results.hydrodynamics.velocities,
                water_levels=results.hydrodynamics.water_levels,
                froude_numbers=results.hydrodynamics.froude_numbers,
            )
        
        # 导出水质结果
        if results.water_quality is not None:
            wq_data = {
                'times': results.water_quality.times,
                'stations': results.water_quality.stations,
            }
            wq_data.update(results.water_quality.concentrations)
            np.savez(os.path.join(output_dir, 'water_quality.npz'), **wq_data)
        
        # 导出边坡稳定性结果
        if results.slope_stability is not None:
            np.savez(
                os.path.join(output_dir, 'slope_stability.npz'),
                times=results.slope_stability.times,
                stations=results.slope_stability.stations,
                sliding_factors=results.slope_stability.sliding_factors,
                overturning_factors=results.slope_stability.overturning_factors,
                uplift_factors=results.slope_stability.uplift_factors,
                seepage_factors=results.slope_stability.seepage_factors,
                comprehensive_factors=results.slope_stability.comprehensive_factors,
                stability_status=results.slope_stability.stability_status,
                channel_stability_index=results.slope_stability.channel_stability_index,
                unstable_section_count=results.slope_stability.unstable_section_count,
            )
        
        print(f"结果已导出到: {output_dir}")

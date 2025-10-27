"""
明渠边坡稳定性系统测试

测试各模块的基本功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from channel_stability.core.channel_system import (
    ChannelSystem,
    ChannelSection,
    SoilProperties,
    LiningProperties,
)
from channel_stability.core.monitoring_network import (
    MonitoringNetwork,
    MonitoringStation,
    StationType,
)
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig


def create_test_channel():
    """创建测试用明渠"""
    soil = SoilProperties(
        cohesion=15.0,
        friction_angle=30.0,
        unit_weight=18.0,
        saturated_unit_weight=20.0,
        permeability=1e-5,
        porosity=0.35,
    )
    
    lining = LiningProperties(
        thickness=0.1,
        density=2400.0,
        elastic_modulus=30.0,
        friction_coeff=0.4,
    )
    
    channel = ChannelSystem(
        name="测试明渠",
        total_length=5000.0,
        sections=[],
    )
    
    base_section = ChannelSection(
        station=0.0,
        bed_elevation=100.0,
        bottom_width=5.0,
        side_slope=2.0,
        max_depth=4.0,
        manning_n=0.025,
        slope_height=3.0,
        slope_angle=26.57,
        soil_properties=soil,
        lining_properties=lining,
    )
    
    channel.create_uniform_sections(
        num_sections=11,
        base_section=base_section,
        bed_slope=0.0001,
    )
    
    return channel


def create_test_monitoring_network():
    """创建测试监测网络"""
    network = MonitoringNetwork(network_name="测试网络")
    network.create_uniform_groundwater_network(
        channel_length=5000.0,
        spacing=500.0,
    )
    network.add_boundary_stations(0.0, 5000.0)
    
    # 添加模拟数据
    times = np.linspace(0, 3600, 10)
    for station in network.groundwater_stations:
        for t in times:
            station.add_observation(
                time=t,
                groundwater_level=98.0 + 0.1 * np.sin(2 * np.pi * t / 3600),
            )
    
    for station in network.rainfall_stations:
        for t in times:
            station.add_observation(
                time=t,
                rainfall=5.0,
            )
    
    return network


class TestChannelSystem:
    """测试明渠系统"""
    
    def test_channel_creation(self):
        """测试明渠创建"""
        channel = create_test_channel()
        assert channel.num_sections == 11
        assert channel.total_length == 5000.0
        assert len(channel.stations) == 11
    
    def test_section_geometry(self):
        """测试断面几何计算"""
        channel = create_test_channel()
        section = channel.sections[0]
        
        depth = 2.0
        area = section.area(depth)
        wp = section.wetted_perimeter(depth)
        R = section.hydraulic_radius(depth)
        B = section.top_width(depth)
        
        assert area > 0
        assert wp > 0
        assert R > 0
        assert B > section.bottom_width


class TestMonitoringNetwork:
    """测试监测网络"""
    
    def test_network_creation(self):
        """测试网络创建"""
        network = create_test_monitoring_network()
        assert network.num_stations > 0
        assert len(network.groundwater_stations) > 0
        assert len(network.rainfall_stations) > 0
    
    def test_data_interpolation(self):
        """测试数据插值"""
        network = create_test_monitoring_network()
        station = network.groundwater_stations[0]
        
        level = station.interpolate_groundwater_level(1800.0)
        assert level is not None
        assert level > 0


class TestHydrodynamics:
    """测试水动力学模块"""
    
    def test_preissmann_solver(self):
        """测试Preissmann求解器"""
        channel = create_test_channel()
        bc = BoundaryConditions.create_constant_bc(
            upstream_q=10.0,
            downstream_h=2.0,
        )
        
        solver = PreissmannHydrodynamicSolver(
            channel=channel,
            boundary_conditions=bc,
            enable_adaptive_mesh=False,
        )
        
        results = solver.solve(
            total_time=600.0,  # 10分钟
            dt=10.0,
            initial_depth=2.0,
            save_interval=5,
        )
        
        assert len(results.times) > 0
        assert len(results.stations) == channel.num_sections
        assert results.depths.shape[1] == channel.num_sections
        assert np.all(results.depths > 0)


class TestWaterQuality:
    """测试水质模块"""
    
    def test_water_quality_solver(self):
        """测试水质求解器"""
        # 先运行水动力学
        channel = create_test_channel()
        bc = BoundaryConditions.create_constant_bc(upstream_q=10.0, downstream_h=2.0)
        
        hydro_solver = PreissmannHydrodynamicSolver(
            channel=channel,
            boundary_conditions=bc,
            enable_adaptive_mesh=False,
        )
        
        hydro_results = hydro_solver.solve(
            total_time=600.0,
            dt=10.0,
            initial_depth=2.0,
            save_interval=5,
        )
        
        # 水质模拟
        wq_params = WaterQualityParameters()
        wq_solver = WaterQualitySolver(parameters=wq_params)
        
        wq_results = wq_solver.solve(
            hydrodynamic_results=hydro_results,
            indicators=['DO', 'BOD'],
            dt=10.0,
        )
        
        assert 'DO' in wq_results.concentrations
        assert 'BOD' in wq_results.concentrations
        assert wq_results.concentrations['DO'].shape == hydro_results.depths.shape
        assert np.all(wq_results.concentrations['DO'] >= 0)


class TestSlopeStability:
    """测试边坡稳定性模块"""
    
    def test_stability_calculator(self):
        """测试稳定性计算器"""
        channel = create_test_channel()
        network = create_test_monitoring_network()
        bc = BoundaryConditions.create_constant_bc(upstream_q=10.0, downstream_h=2.0)
        
        # 水动力学计算
        hydro_solver = PreissmannHydrodynamicSolver(
            channel=channel,
            boundary_conditions=bc,
            enable_adaptive_mesh=False,
        )
        
        hydro_results = hydro_solver.solve(
            total_time=600.0,
            dt=10.0,
            initial_depth=2.0,
            save_interval=5,
        )
        
        # 稳定性计算
        stability_calc = SlopeStabilityCalculator(
            channel=channel,
            monitoring_network=network,
        )
        
        stability_results = stability_calc.compute_stability(
            hydrodynamic_results=hydro_results,
            time_indices=[0, len(hydro_results.times)//2, -1],
        )
        
        assert len(stability_results.times) > 0
        assert stability_results.comprehensive_factors.shape[1] == channel.num_sections
        assert np.all(stability_results.comprehensive_factors > 0)


class TestIntegratedSimulation:
    """测试综合仿真系统"""
    
    def test_integrated_simulator(self):
        """测试综合仿真器"""
        channel = create_test_channel()
        network = create_test_monitoring_network()
        bc = BoundaryConditions.create_constant_bc(upstream_q=10.0, downstream_h=2.0)
        
        simulator = IntegratedSimulator(
            channel=channel,
            monitoring_network=network,
            boundary_conditions=bc,
        )
        
        config = SimulationConfig(
            total_time=600.0,
            dt=10.0,
            save_interval=5,
            enable_hydrodynamics=True,
            enable_water_quality=True,
            enable_slope_stability=True,
            simulated_indicators=['DO', 'BOD'],
            stability_time_interval=10,
        )
        
        results = simulator.run_simulation(config)
        
        assert results.hydrodynamics is not None
        assert results.water_quality is not None
        assert results.slope_stability is not None
        
        # 测试摘要
        summary = results.summary()
        assert 'hydrodynamics' in summary
        assert 'water_quality' in summary
        assert 'slope_stability' in summary


def test_reduced_order_models():
    """测试降阶模型"""
    from channel_stability.reduced_order.linear_reduced_model import (
        LinearizedHydrodynamics,
        LinearizedWaterQuality,
        LinearizedSlopeStability,
    )
    
    # 线性化水动力学
    stations = np.linspace(0, 5000, 11)
    bed_elevs = 100.0 - 0.0001 * stations
    widths = np.full(11, 5.0)
    
    linear_hydro = LinearizedHydrodynamics(
        stations=stations,
        bed_elevations=bed_elevs,
        bottom_widths=widths,
    )
    
    upstream_q = np.full(100, 10.0)
    result = linear_hydro.solve_kinematic_wave(
        total_time=1000.0,
        dt=10.0,
        upstream_discharge_series=upstream_q,
    )
    
    assert result.times is not None
    assert result.values is not None
    assert np.all(result.values > 0)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])

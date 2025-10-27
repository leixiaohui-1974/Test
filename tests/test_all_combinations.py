"""
全面组合测试

测试所有可能的单独模块和组合场景
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def test_case_1_hydrodynamics_only():
    """测试用例1: 仅水动力学"""
    print("\n" + "="*70)
    print("测试用例1: 仅水动力学模拟")
    print("="*70)
    
    from channel_stability.core.channel_system import ChannelSystem, ChannelSection, SoilProperties
    from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
    from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
    
    # 创建简单系统
    soil = SoilProperties(15.0, 30.0, 18.0, 20.0, 1e-5, 0.35)
    channel = ChannelSystem(name="测试", total_length=500.0, sections=[])
    
    for i in range(3):
        section = ChannelSection(
            station=i*250.0, bed_elevation=100.0-0.0001*i*250.0,
            bottom_width=5.0, side_slope=2.0, max_depth=4.0, manning_n=0.025,
            slope_height=3.0, slope_angle=26.57, soil_properties=soil
        )
        channel.sections.append(section)
    
    bc = BoundaryConditions.create_constant_bc(upstream_q=10.0, downstream_h=2.0)
    solver = PreissmannHydrodynamicSolver(channel, bc, enable_adaptive_mesh=False)
    
    # 单步仿真
    results = solver.solve(total_time=10.0, dt=10.0, initial_depth=2.0, save_interval=1)
    
    print(f"✓ 水动力学仿真完成")
    print(f"  时间步数: {len(results.times)}")
    print(f"  水深范围: {np.min(results.depths):.3f} - {np.max(results.depths):.3f} m")
    print(f"  流量范围: {np.min(results.discharges):.3f} - {np.max(results.discharges):.3f} m³/s")
    
    assert np.all(results.depths > 0) and np.all(results.depths < 10), "水深应在合理范围"
    return results


def test_case_2_water_quality_only():
    """测试用例2: 仅水质模拟（使用固定水动力场）"""
    print("\n" + "="*70)
    print("测试用例2: 仅水质模拟")
    print("="*70)
    
    from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
    from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
    from channel_stability.hydrodynamics.preissmann_solver import HydrodynamicResults
    
    # 创建固定水动力场
    times = np.linspace(0, 600, 11)
    stations = np.linspace(0, 500, 3)
    depths = np.full((11, 3), 2.0)
    discharges = np.full((11, 3), 10.0)
    velocities = np.full((11, 3), 0.5)
    water_levels = 100.0 + depths
    froude_numbers = np.full((11, 3), 0.2)
    
    hydro_results = HydrodynamicResults(
        times, stations, depths, discharges, velocities, water_levels, froude_numbers
    )
    
    params = WaterQualityParameters(temperature=20.0)
    solver = WaterQualitySolver(parameters=params)
    
    wq_results = solver.solve(hydro_results, ['DO', 'BOD'], dt=60.0)
    
    print(f"✓ 水质模拟完成")
    print(f"  时间步数: {len(wq_results.times)}")
    for ind in ['DO', 'BOD']:
        C = wq_results.concentrations[ind]
        print(f"  {ind}: {np.min(C):.2f} - {np.max(C):.2f} mg/L")
        assert np.all(C >= 0), f"{ind}应非负"
    
    return wq_results


def test_case_3_slope_stability_only():
    """测试用例3: 仅边坡稳定性（使用固定水位和地下水位）"""
    print("\n" + "="*70)
    print("测试用例3: 仅边坡稳定性分析")
    print("="*70)
    
    from channel_stability.core.channel_system import ChannelSystem, ChannelSection, SoilProperties, LiningProperties
    from channel_stability.core.monitoring_network import MonitoringNetwork, MonitoringStation, StationType
    from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
    from channel_stability.hydrodynamics.preissmann_solver import HydrodynamicResults
    
    soil = SoilProperties(15.0, 30.0, 18.0, 20.0, 1e-5, 0.35)
    lining = LiningProperties(0.1, 2400.0, 30.0, 0.4)
    
    channel = ChannelSystem(name="测试", total_length=500.0, sections=[])
    for i in range(3):
        section = ChannelSection(
            station=i*250.0, bed_elevation=100.0, bottom_width=5.0, side_slope=2.0,
            max_depth=4.0, manning_n=0.025, slope_height=3.0, slope_angle=26.57,
            soil_properties=soil, lining_properties=lining
        )
        channel.sections.append(section)
    
    # 监测网络
    network = MonitoringNetwork(network_name="测试")
    for i, sec in enumerate(channel.sections):
        station = MonitoringStation(f"GW_{i}", StationType.GROUNDWATER, sec.station)
        for t in [0, 300, 600]:
            station.add_observation(time=t, groundwater_level=98.0)
        network.stations.append(station)
    
    # 固定水动力场
    hydro_results = HydrodynamicResults(
        times=np.array([0, 300, 600]),
        stations=channel.stations,
        depths=np.full((3, 3), 2.0),
        discharges=np.full((3, 3), 10.0),
        velocities=np.full((3, 3), 0.5),
        water_levels=np.full((3, 3), 102.0),
        froude_numbers=np.full((3, 3), 0.2),
    )
    
    calculator = SlopeStabilityCalculator(channel, network)
    stab_results = calculator.compute_stability(hydro_results)
    
    print(f"✓ 边坡稳定性计算完成")
    print(f"  时间点数: {len(stab_results.times)}")
    print(f"  稳定系数范围: {np.min(stab_results.comprehensive_factors):.2f} - {np.max(stab_results.comprehensive_factors):.2f}")
    print(f"  不稳定断面数: {int(np.max(stab_results.unstable_section_count))}")
    
    assert np.all(stab_results.comprehensive_factors > 0), "稳定系数应为正"
    assert np.all(stab_results.comprehensive_factors <= 10.0), "稳定系数应≤10"
    
    return stab_results


def test_case_4_hydrodynamics_water_quality():
    """测试用例4: 水动力学 + 水质"""
    print("\n" + "="*70)
    print("测试用例4: 水动力学 + 水质组合")
    print("="*70)
    
    from channel_stability.core.channel_system import ChannelSystem, ChannelSection, SoilProperties
    from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
    from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
    from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
    from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
    
    soil = SoilProperties(15.0, 30.0, 18.0, 20.0, 1e-5, 0.35)
    channel = ChannelSystem(name="测试", total_length=500.0, sections=[])
    for i in range(3):
        section = ChannelSection(
            station=i*250.0, bed_elevation=100.0, bottom_width=5.0, side_slope=2.0,
            max_depth=4.0, manning_n=0.025, slope_height=3.0, slope_angle=26.57,
            soil_properties=soil
        )
        channel.sections.append(section)
    
    bc = BoundaryConditions.create_constant_bc(upstream_q=10.0, downstream_h=2.0)
    
    # 水动力学
    hydro_solver = PreissmannHydrodynamicSolver(channel, bc, enable_adaptive_mesh=False)
    hydro_results = hydro_solver.solve(total_time=60.0, dt=30.0, initial_depth=2.0, save_interval=1)
    
    # 水质
    wq_params = WaterQualityParameters()
    wq_solver = WaterQualitySolver(parameters=wq_params)
    wq_results = wq_solver.solve(hydro_results, ['DO'], dt=30.0)
    
    print(f"✓ 水动力学+水质组合完成")
    print(f"  水深: {np.min(hydro_results.depths):.2f} - {np.max(hydro_results.depths):.2f} m")
    print(f"  DO: {np.min(wq_results.concentrations['DO']):.2f} - {np.max(wq_results.concentrations['DO']):.2f} mg/L")
    
    assert len(hydro_results.times) == len(wq_results.times), "时间序列应一致"
    return hydro_results, wq_results


def test_case_5_full_chain():
    """测试用例5: 完整链路（水动力学 → 水质 → 边坡稳定性）"""
    print("\n" + "="*70)
    print("测试用例5: 完整链路测试")
    print("="*70)
    
    from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig
    from channel_stability.core.channel_system import ChannelSystem, ChannelSection, SoilProperties, LiningProperties
    from channel_stability.core.monitoring_network import MonitoringNetwork, MonitoringStation, StationType
    from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
    
    # 创建系统
    soil = SoilProperties(15.0, 30.0, 18.0, 20.0, 1e-5, 0.35)
    lining = LiningProperties(0.1, 2400.0, 30.0, 0.4)
    channel = ChannelSystem(name="测试", total_length=500.0, sections=[])
    
    for i in range(3):
        section = ChannelSection(
            station=i*250.0, bed_elevation=100.0, bottom_width=5.0, side_slope=2.0,
            max_depth=4.0, manning_n=0.025, slope_height=3.0, slope_angle=26.57,
            soil_properties=soil, lining_properties=lining
        )
        channel.sections.append(section)
    
    network = MonitoringNetwork(network_name="测试")
    for i, sec in enumerate(channel.sections):
        station = MonitoringStation(f"GW_{i}", StationType.GROUNDWATER, sec.station)
        for t in np.linspace(0, 60, 5):
            station.add_observation(time=t, groundwater_level=98.0 + 0.1*np.sin(t/10))
        network.stations.append(station)
    
    bc = BoundaryConditions.create_constant_bc(upstream_q=10.0, downstream_h=2.0)
    
    simulator = IntegratedSimulator(channel, network, bc)
    config = SimulationConfig(
        total_time=60.0,
        dt=30.0,
        save_interval=1,
        enable_hydrodynamics=True,
        enable_water_quality=True,
        enable_slope_stability=True,
        simulated_indicators=['DO'],
        stability_time_interval=1,
    )
    
    results = simulator.run_simulation(config)
    
    print(f"✓ 完整链路仿真完成")
    summary = results.summary()
    
    if 'hydrodynamics' in summary:
        print(f"  水动力学: 最大水深={summary['hydrodynamics']['max_depth']:.2f}m")
    if 'water_quality' in summary:
        print(f"  水质: DO={summary['water_quality']['DO_min']:.2f}-{summary['water_quality']['DO_max']:.2f}mg/L")
    if 'slope_stability' in summary:
        print(f"  边坡稳定性: 最小系数={summary['slope_stability']['min_stability_factor']:.2f}")
    
    return results


def test_case_6_flood_scenario():
    """测试用例6: 洪水场景（变化边界条件）"""
    print("\n" + "="*70)
    print("测试用例6: 洪水过程场景")
    print("="*70)
    
    from channel_stability.core.channel_system import ChannelSystem, ChannelSection, SoilProperties, LiningProperties
    from channel_stability.core.monitoring_network import MonitoringNetwork, MonitoringStation, StationType
    from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
    from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
    from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
    
    soil = SoilProperties(15.0, 30.0, 18.0, 20.0, 1e-5, 0.35)
    lining = LiningProperties(0.1, 2400.0, 30.0, 0.4)
    channel = ChannelSystem(name="洪水测试", total_length=500.0, sections=[])
    
    for i in range(3):
        section = ChannelSection(
            station=i*250.0, bed_elevation=100.0, bottom_width=5.0, side_slope=2.0,
            max_depth=4.0, manning_n=0.025, slope_height=3.0, slope_angle=26.57,
            soil_properties=soil, lining_properties=lining
        )
        channel.sections.append(section)
    
    # 洪水过程
    flood_series = [(0, 5.0), (30, 15.0), (60, 5.0)]
    bc = BoundaryConditions.create_hydrograph_bc(flood_series, downstream_h=2.0)
    
    # 水动力学
    solver = PreissmannHydrodynamicSolver(channel, bc, enable_adaptive_mesh=False)
    hydro_results = solver.solve(total_time=60.0, dt=30.0, initial_depth=2.0, save_interval=1)
    
    # 监测网络
    network = MonitoringNetwork(network_name="洪水测试")
    for i, sec in enumerate(channel.sections):
        station = MonitoringStation(f"GW_{i}", StationType.GROUNDWATER, sec.station)
        for t in hydro_results.times:
            station.add_observation(time=t, groundwater_level=98.0)
        network.stations.append(station)
    
    # 边坡稳定性
    calculator = SlopeStabilityCalculator(channel, network)
    stab_results = calculator.compute_stability(hydro_results)
    
    print(f"✓ 洪水场景仿真完成")
    print(f"  上游流量: 5.0 → 15.0 → 5.0 m³/s")
    print(f"  水深变化: {np.min(hydro_results.depths):.2f} - {np.max(hydro_results.depths):.2f} m")
    print(f"  稳定系数: {np.min(stab_results.comprehensive_factors):.2f} - {np.max(stab_results.comprehensive_factors):.2f}")
    
    return hydro_results, stab_results


def test_case_7_rainfall_scenario():
    """测试用例7: 降雨场景（地下水位变化）"""
    print("\n" + "="*70)
    print("测试用例7: 降雨影响场景")
    print("="*70)
    
    from channel_stability.slope_stability.failure_mechanisms import LiningStabilityAnalysis
    from channel_stability.core.channel_system import SoilProperties, LiningProperties
    
    soil = SoilProperties(15.0, 30.0, 18.0, 20.0, 1e-5, 0.35)
    lining = LiningProperties(0.1, 2400.0, 30.0, 0.4)
    
    rainfalls = [0, 10, 20, 30, 40, 50]
    results = []
    
    print("\n降雨强度(mm/h) | 滑动系数 | 综合系数")
    print("-" * 50)
    
    for rain in rainfalls:
        factors = LiningStabilityAnalysis.compute_comprehensive_stability(
            26.57, 3.0, soil, lining, 1.0, 2.0, rain
        )
        results.append(factors.comprehensive_factor)
        print(f"{rain:14d} | {factors.sliding_factor:10.2f} | {factors.comprehensive_factor:10.2f}")
    
    print(f"\n✓ 降雨场景分析完成")
    print(f"  稳定系数变化: {min(results):.2f} - {max(results):.2f}")
    
    return results


def run_all_combination_tests():
    """运行所有组合测试"""
    print("\n" + "#"*70)
    print("# 全面组合测试套件")
    print("#"*70)
    
    test_results = {}
    
    try:
        # 单独模块测试
        print("\n【单独模块测试】")
        test_results['case1_hydro'] = test_case_1_hydrodynamics_only()
        test_results['case2_wq'] = test_case_2_water_quality_only()
        test_results['case3_slope'] = test_case_3_slope_stability_only()
        
        # 两模块组合测试
        print("\n【两模块组合测试】")
        test_results['case4_hydro_wq'] = test_case_4_hydrodynamics_water_quality()
        
        # 完整链路测试
        print("\n【三模块完整链路测试】")
        test_results['case5_full'] = test_case_5_full_chain()
        
        # 场景测试
        print("\n【场景测试】")
        test_results['case6_flood'] = test_case_6_flood_scenario()
        test_results['case7_rainfall'] = test_case_7_rainfall_scenario()
        
        print("\n" + "#"*70)
        print("# 测试结果汇总")
        print("#"*70)
        
        print("\n✓ 测试用例1: 仅水动力学 - 通过")
        print("✓ 测试用例2: 仅水质 - 通过")
        print("✓ 测试用例3: 仅边坡稳定性 - 通过")
        print("✓ 测试用例4: 水动力学+水质 - 通过")
        print("✓ 测试用例5: 完整链路 - 通过")
        print("✓ 测试用例6: 洪水场景 - 通过")
        print("✓ 测试用例7: 降雨场景 - 通过")
        
        print("\n" + "="*70)
        print("所有组合测试通过！ ✓✓✓ (7/7)")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 组合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_combination_tests()
    exit(0 if success else 1)

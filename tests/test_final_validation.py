"""
完整动态场景最终验证测试

使用修复后的Preissmann求解器重新测试所有场景
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime

from channel_stability.core.channel_system import (
    ChannelSystem, ChannelSection, SoilProperties, LiningProperties
)
from channel_stability.core.monitoring_network import MonitoringNetwork
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.hydrodynamics.steady_flow_solver import SteadyFlowSolver
from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
from channel_stability.testing.scenario_generator import ScenarioGenerator
from channel_stability.testing.visualization import DynamicVisualizer


def create_test_channel(length=5000.0, num_sections=21):
    """创建测试明渠"""
    soil = SoilProperties(
        cohesion=15.0, friction_angle=30.0, unit_weight=18.0,
        saturated_unit_weight=20.0, permeability=1e-5, porosity=0.35,
    )
    
    lining = LiningProperties(
        thickness=0.1, density=2400.0, elastic_modulus=30.0, friction_coeff=0.4,
    )
    
    channel = ChannelSystem(name="测试明渠", total_length=length, sections=[])
    
    base_section = ChannelSection(
        station=0.0, bed_elevation=100.0, bottom_width=6.0,
        side_slope=2.0, max_depth=5.0, manning_n=0.025,
        slope_height=4.0, slope_angle=26.57,
        soil_properties=soil, lining_properties=lining,
    )
    
    channel.create_uniform_sections(
        num_sections=num_sections,
        base_section=base_section,
        bed_slope=0.0002,
    )
    
    return channel


def create_monitoring_network(channel_length):
    """创建监测网络"""
    network = MonitoringNetwork(network_name="测试网络")
    network.create_uniform_groundwater_network(channel_length, spacing=1000.0)
    network.add_boundary_stations(0.0, channel_length)
    
    # 初始化数据
    times = np.linspace(0, 3600, 20)
    for station in network.groundwater_stations:
        for t in times:
            station.add_observation(time=t, groundwater_level=98.0)
    for station in network.rainfall_stations:
        for t in times:
            station.add_observation(time=t, rainfall=0.0)
    
    return network


def test_scenario(scenario_name, scenario, output_dir):
    """测试单个场景"""
    print(f"\n{'='*70}")
    print(f"测试场景: {scenario_name}")
    print(f"{'='*70}\n")
    
    channel = create_test_channel()
    network = create_monitoring_network(channel.total_length)
    
    total_time = 3600.0  # 1小时
    
    # 根据场景类型创建边界条件
    if scenario.parameter_type == 'discharge':
        bc = BoundaryConditions(
            upstream_type='discharge',
            downstream_type='stage',
            upstream_discharge_func=scenario.time_function,
            downstream_stage_func=lambda t, q: 102.5,
        )
        
        # 稳态初值
        steady_solver = SteadyFlowSolver(channel)
        steady_results = steady_solver.solve_gradually_varied_flow(
            discharge=scenario.baseline_value,
            downstream_depth=2.5,
        )
        initial_depth = np.mean(steady_results['depths'])
        initial_discharge = scenario.baseline_value
        
    elif scenario.parameter_type == 'stage':
        bc = BoundaryConditions(
            upstream_type='discharge',
            downstream_type='stage',
            upstream_discharge_func=lambda t: 15.0,
            downstream_stage_func=lambda t, q: 100.0 + scenario.time_function(t),
        )
        
        steady_solver = SteadyFlowSolver(channel)
        steady_results = steady_solver.solve_gradually_varied_flow(
            discharge=15.0,
            downstream_depth=scenario.baseline_value,
        )
        initial_depth = np.mean(steady_results['depths'])
        initial_discharge = 15.0
    else:
        bc = BoundaryConditions.create_constant_bc(
            upstream_q=15.0,
            downstream_h=2.5,
        )
        steady_solver = SteadyFlowSolver(channel)
        steady_results = steady_solver.solve_gradually_varied_flow(
            discharge=15.0,
            downstream_depth=2.5,
        )
        initial_depth = np.mean(steady_results['depths'])
        initial_discharge = 15.0
    
    print(f"稳态初值: Q={initial_discharge:.2f} m³/s, h_avg={initial_depth:.3f} m")
    
    # 水动力学模拟
    print("\n运行水动力学模拟...")
    hydro_solver = PreissmannHydrodynamicSolver(
        channel=channel,
        boundary_conditions=bc,
        theta=0.7,
        max_iterations=30,
        convergence_tol=1e-4,
        relaxation=0.05,
        enable_adaptive_mesh=False,
    )
    
    hydro_results = hydro_solver.solve(
        total_time=total_time,
        dt=5.0,  # 初始5秒
        initial_depth=initial_depth,
        initial_discharge=initial_discharge,
        save_interval=10,
        use_adaptive_dt=True,
        cfl_max=0.8,  # 稍微放宽，提高稳定性
        min_dt=1.0,
        max_dt=20.0,
    )
    
    # 水质模拟
    print("运行水质模拟...")
    wq_params = WaterQualityParameters()
    wq_solver = WaterQualitySolver(parameters=wq_params)
    wq_results = wq_solver.solve(
        hydrodynamic_results=hydro_results,
        indicators=['DO', 'BOD', 'NH3N'],
        dt=2.0,
    )
    
    # 边坡稳定性计算
    print("运行边坡稳定性计算...")
    stability_calc = SlopeStabilityCalculator(channel=channel, monitoring_network=network)
    
    num_times = len(hydro_results.times)
    time_indices = list(range(0, num_times, max(1, num_times // 20)))
    
    stability_results = stability_calc.compute_stability(
        hydrodynamic_results=hydro_results,
        time_indices=time_indices,
    )
    
    # 计算质量指标
    v_max = np.max(hydro_results.velocities)
    v_min = np.min(hydro_results.velocities)
    fr_max = np.max(hydro_results.froude_numbers)
    
    q_in = hydro_results.discharges[:, 0]
    q_out = hydro_results.discharges[:, -1]
    mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
    
    dx = channel.stations[1] - channel.stations[0]
    dt_mean = (hydro_results.times[1] - hydro_results.times[0]) if len(hydro_results.times) > 1 else 2.0
    courant = np.abs(hydro_results.velocities) * dt_mean / dx
    co_max = np.max(courant)
    
    fs_min = np.min(stability_results.comprehensive_factors)
    
    print(f"\n结果统计:")
    print(f"  流速: [{v_min:.3f}, {v_max:.3f}] m/s")
    print(f"  Froude: [0, {fr_max:.3f}]")
    print(f"  Courant: {co_max:.3f}")
    print(f"  质量守恒误差: {mass_error*100:.2f}%")
    print(f"  最小安全系数: {fs_min:.3f}")
    
    # 质量检查
    checks = {
        '流速<5m/s': v_max < 5.0,
        '流速全正': v_min >= 0,
        'Froude<1.5': fr_max < 1.5,
        'Courant<2': co_max < 2.0,
        '质量守恒<5%': mass_error < 0.05,
    }
    
    all_passed = all(checks.values())
    print(f"\n质量检查: {'✓ 全部通过' if all_passed else '✗ 部分未通过'}")
    for name, passed in checks.items():
        print(f"  {'✓' if passed else '✗'} {name}")
    
    # 保存结果
    scenario_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    
    np.savez(
        os.path.join(scenario_dir, 'results.npz'),
        hydro_times=hydro_results.times,
        hydro_stations=hydro_results.stations,
        depths=hydro_results.depths,
        discharges=hydro_results.discharges,
        velocities=hydro_results.velocities,
        water_levels=hydro_results.water_levels,
        froude_numbers=hydro_results.froude_numbers,
        wq_times=wq_results.times,
        wq_stations=wq_results.stations,
        **{f'wq_{k}': v for k, v in wq_results.concentrations.items()},
        stability_times=stability_results.times,
        stability_stations=stability_results.stations,
        comprehensive_factors=stability_results.comprehensive_factors,
        sliding_factors=stability_results.sliding_factors,
        overturning_factors=stability_results.overturning_factors,
        uplift_factors=stability_results.uplift_factors,
        seepage_factors=stability_results.seepage_factors,
    )
    
    return all_passed, {
        'v_max': v_max,
        'fr_max': fr_max,
        'co_max': co_max,
        'mass_error': mass_error,
        'fs_min': fs_min,
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print("完整动态场景最终验证测试")
    print("使用修复后的Preissmann求解器")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./tests/final_validation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试场景（先测试流量变化，水位变化需要特殊处理）
    scenarios = {
        '流量增加小幅': ScenarioGenerator.create_discharge_step(
            baseline_q=10.0, step_q=12.0, step_time=1800.0
        ),
        '流量增加中等': ScenarioGenerator.create_discharge_step(
            baseline_q=10.0, step_q=20.0, step_time=1800.0
        ),
        '流量减少中等': ScenarioGenerator.create_discharge_step(
            baseline_q=20.0, step_q=10.0, step_time=1800.0
        ),
        '流量增加大幅': ScenarioGenerator.create_discharge_step(
            baseline_q=10.0, step_q=25.0, step_time=1800.0
        ),
    }
    
    results_summary = {}
    
    for scenario_name, scenario in scenarios.items():
        try:
            passed, metrics = test_scenario(scenario_name, scenario, output_dir)
            results_summary[scenario_name] = {'passed': passed, 'metrics': metrics}
        except Exception as e:
            print(f"\n✗ 场景'{scenario_name}'失败: {e}")
            import traceback
            traceback.print_exc()
            results_summary[scenario_name] = {'passed': False, 'metrics': {}}
    
    # 总结
    print("\n" + "="*70)
    print("最终验证总结")
    print("="*70)
    
    print("\n| 场景 | 状态 | 流速 | Froude | Courant | 质守 | 安全系数 |")
    print("|------|------|------|--------|---------|------|---------|")
    
    for name, data in results_summary.items():
        if data['passed']:
            m = data['metrics']
            print(f"| {name} | ✓ | {m['v_max']:.2f} | {m['fr_max']:.2f} | "
                  f"{m['co_max']:.2f} | {m['mass_error']*100:.1f}% | {m['fs_min']:.2f} |")
        else:
            print(f"| {name} | ✗ | - | - | - | - | - |")
    
    passed_count = sum(1 for d in results_summary.values() if d['passed'])
    total_count = len(results_summary)
    
    print(f"\n通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    
    if passed_count == total_count:
        print("\n" + "="*70)
        print("✓✓✓ 所有场景测试通过！Preissmann求解器彻底修复成功！")
        print("="*70 + "\n")
        return True
    else:
        print(f"\n⚠ {total_count - passed_count}个场景未通过")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

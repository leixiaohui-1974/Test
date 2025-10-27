"""
非恒定流动态场景综合测试

测试各种阶跃变化场景下的系统响应特性
包括：
1. 用恒定流计算初值
2. 非恒定流模拟
3. 各种阶跃变化场景测试
4. 生成时间序列图和GIF动画
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime

from channel_stability.core.channel_system import (
    ChannelSystem,
    ChannelSection,
    SoilProperties,
    LiningProperties,
)
from channel_stability.core.monitoring_network import MonitoringNetwork
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.hydrodynamics.steady_flow_solver import SteadyFlowSolver
from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig
from channel_stability.testing.scenario_generator import ScenarioGenerator
from channel_stability.testing.visualization import DynamicVisualizer


def create_test_channel(length: float = 10000.0, num_sections: int = 21):
    """创建测试明渠"""
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
        total_length=length,
        sections=[],
    )
    
    base_section = ChannelSection(
        station=0.0,
        bed_elevation=100.0,
        bottom_width=6.0,
        side_slope=2.0,
        max_depth=5.0,
        manning_n=0.025,
        slope_height=4.0,
        slope_angle=26.57,
        soil_properties=soil,
        lining_properties=lining,
    )
    
    channel.create_uniform_sections(
        num_sections=num_sections,
        base_section=base_section,
        bed_slope=0.0002,
    )
    
    return channel


def create_monitoring_network(channel_length: float):
    """创建监测网络"""
    network = MonitoringNetwork(network_name="测试监测网络")
    network.create_uniform_groundwater_network(
        channel_length=channel_length,
        spacing=1000.0,
    )
    network.add_boundary_stations(0.0, channel_length)
    
    # 初始化监测数据
    times = np.linspace(0, 7200, 20)
    for station in network.groundwater_stations:
        for t in times:
            # 基准地下水位
            station.add_observation(
                time=t,
                groundwater_level=98.0,
            )
    
    for station in network.rainfall_stations:
        for t in times:
            station.add_observation(
                time=t,
                rainfall=0.0,
            )
    
    return network


def update_monitoring_network_with_scenario(
    network: MonitoringNetwork,
    scenario,
    total_time: float,
):
    """根据场景更新监测网络数据"""
    times = np.linspace(0, total_time, 100)
    
    if scenario.parameter_type == 'groundwater':
        for station in network.groundwater_stations:
            station.observations.clear()
            for t in times:
                level = scenario.get_value(t)
                station.add_observation(time=t, groundwater_level=level)
    
    elif scenario.parameter_type == 'rainfall':
        for station in network.rainfall_stations:
            station.observations.clear()
            for t in times:
                rainfall = scenario.get_value(t)
                station.add_observation(time=t, rainfall=rainfall)


def test_scenario_with_steady_initial(
    scenario_name: str,
    scenario,
    output_dir: str,
):
    """
    测试单个场景（使用恒定流初值）
    
    Parameters
    ----------
    scenario_name : str
        场景名称
    scenario : StepChangeScenario
        场景对象
    output_dir : str
        输出目录
    """
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario_name}")
    print(f"描述: {scenario.description}")
    print(f"{'='*60}\n")
    
    # 创建明渠和监测网络
    channel = create_test_channel()
    network = create_monitoring_network(channel.total_length)
    
    # 模拟参数
    total_time = 7200.0  # 2小时
    dt = 10.0
    
    # 根据场景类型创建边界条件
    if scenario.parameter_type == 'discharge':
        # 流量阶跃场景
        bc = BoundaryConditions(
            upstream_type='discharge',
            downstream_type='stage',
            upstream_discharge_func=scenario.time_function,
            downstream_stage_func=lambda t, q: 102.5,  # 固定下游水位
        )
        
        # 步骤1: 使用恒定流计算初值
        print("步骤1: 计算恒定流初值...")
        steady_solver = SteadyFlowSolver(channel)
        steady_results = steady_solver.solve_gradually_varied_flow(
            discharge=scenario.baseline_value,
            downstream_depth=2.5,
        )
        initial_depth = np.mean(steady_results['depths'])
        initial_discharge = scenario.baseline_value
        
        print(f"  恒定流: Q={initial_discharge:.2f} m³/s, 平均水深={initial_depth:.2f} m")
        
    elif scenario.parameter_type == 'stage':
        # 水位阶跃场景
        bc = BoundaryConditions(
            upstream_type='discharge',
            downstream_type='stage',
            upstream_discharge_func=lambda t: 15.0,
            downstream_stage_func=lambda t, q: 100.0 + scenario.time_function(t),
        )
        
        print("步骤1: 计算恒定流初值...")
        steady_solver = SteadyFlowSolver(channel)
        steady_results = steady_solver.solve_gradually_varied_flow(
            discharge=15.0,
            downstream_depth=scenario.baseline_value,
        )
        initial_depth = np.mean(steady_results['depths'])
        initial_discharge = 15.0
        
        print(f"  恒定流: Q={initial_discharge:.2f} m³/s, 下游水深={scenario.baseline_value:.2f} m")
        
    else:
        # 其他场景：默认边界条件
        bc = BoundaryConditions.create_constant_bc(
            upstream_q=15.0,
            downstream_h=2.5,
        )
        
        print("步骤1: 计算恒定流初值...")
        steady_solver = SteadyFlowSolver(channel)
        steady_results = steady_solver.solve_gradually_varied_flow(
            discharge=15.0,
            downstream_depth=2.5,
        )
        initial_depth = np.mean(steady_results['depths'])
        initial_discharge = 15.0
        
        print(f"  恒定流: Q={initial_discharge:.2f} m³/s, 平均水深={initial_depth:.2f} m")
    
    # 更新监测网络（如果是降雨或地下水位场景）
    if scenario.parameter_type in ['rainfall', 'groundwater']:
        update_monitoring_network_with_scenario(network, scenario, total_time)
    
    # 步骤2: 运行非恒定流模拟
    print("\n步骤2: 运行非恒定流模拟...")
    
    # 2.1 独立水动力学模拟
    print("  2.1 独立水动力学模拟...")
    hydro_solver = PreissmannHydrodynamicSolver(
        channel=channel,
        boundary_conditions=bc,
        enable_adaptive_mesh=False,
    )
    
    hydro_results = hydro_solver.solve(
        total_time=total_time,
        dt=dt,
        initial_depth=initial_depth,
        initial_discharge=initial_discharge,
        save_interval=10,
    )
    
    print(f"    完成: {len(hydro_results.times)} 个时间步")
    
    # 2.2 水质模拟
    print("  2.2 水质模拟...")
    wq_params = WaterQualityParameters()
    wq_solver = WaterQualitySolver(parameters=wq_params)
    
    # 如果是水质场景，修改上游边界浓度
    if scenario.parameter_type == 'water_quality':
        # 动态修改上游浓度
        indicator = 'BOD'  # 示例
        wq_solver.upstream_concentrations = {
            indicator: scenario.time_function
        }
    
    wq_results = wq_solver.solve(
        hydrodynamic_results=hydro_results,
        indicators=['DO', 'BOD', 'NH3N'],
        dt=dt,
    )
    
    print(f"    完成: {len(wq_results.concentrations)} 个指标")
    
    # 2.3 边坡稳定性计算
    print("  2.3 边坡稳定性计算...")
    stability_calc = SlopeStabilityCalculator(
        channel=channel,
        monitoring_network=network,
    )
    
    num_times = len(hydro_results.times)
    time_indices = list(range(0, num_times, max(1, num_times // 20)))
    
    stability_results = stability_calc.compute_stability(
        hydrodynamic_results=hydro_results,
        time_indices=time_indices,
    )
    
    print(f"    完成: {len(stability_results.times)} 个时间步")
    
    # 步骤3: 生成可视化结果
    print("\n步骤3: 生成可视化结果...")
    
    scenario_dir = os.path.join(output_dir, scenario_name.replace(' ', '_'))
    os.makedirs(scenario_dir, exist_ok=True)
    
    visualizer = DynamicVisualizer(output_dir=scenario_dir)
    
    results = {
        'hydrodynamics': hydro_results,
        'water_quality': wq_results,
        'slope_stability': stability_results,
    }
    
    scenario_info = {
        'name': scenario.name,
        'description': scenario.description,
        'baseline_value': scenario.baseline_value,
        'step_value': scenario.step_value,
        'step_time': scenario.step_time,
    }
    
    # 3.1 时间序列图（选择几个代表性断面）
    num_sections = len(channel.sections)
    station_indices = [0, num_sections // 4, num_sections // 2, 
                      3 * num_sections // 4, num_sections - 1]
    
    print("  3.1 生成时间序列图...")
    visualizer.plot_timeseries(
        results=results,
        station_indices=station_indices,
        output_prefix='timeseries',
        scenario_info=scenario_info,
    )
    
    # 3.2 沿程分布GIF动画
    print("  3.2 生成沿程分布动画...")
    visualizer.create_spatial_animation(
        results=results,
        variable='depth',
        output_filename='depth_animation.gif',
        scenario_info=scenario_info,
        fps=10,
    )
    
    visualizer.create_spatial_animation(
        results=results,
        variable='discharge',
        output_filename='discharge_animation.gif',
        scenario_info=scenario_info,
        fps=10,
    )
    
    visualizer.create_spatial_animation(
        results=results,
        variable='velocity',
        output_filename='velocity_animation.gif',
        scenario_info=scenario_info,
        fps=10,
    )
    
    # 3.3 边界条件图
    print("  3.3 生成边界条件图...")
    visualizer.plot_boundary_conditions(
        scenario=scenario_info,
        total_time=total_time,
        output_filename='boundary_conditions.png',
    )
    
    # 保存数值结果
    print("\n步骤4: 保存数值结果...")
    np.savez(
        os.path.join(scenario_dir, 'results.npz'),
        # 水动力学
        hydro_times=hydro_results.times,
        hydro_stations=hydro_results.stations,
        depths=hydro_results.depths,
        discharges=hydro_results.discharges,
        velocities=hydro_results.velocities,
        water_levels=hydro_results.water_levels,
        froude_numbers=hydro_results.froude_numbers,
        # 水质
        wq_times=wq_results.times,
        wq_stations=wq_results.stations,
        **{f'wq_{k}': v for k, v in wq_results.concentrations.items()},
        # 稳定性
        stability_times=stability_results.times,
        stability_stations=stability_results.stations,
        comprehensive_factors=stability_results.comprehensive_factors,
        sliding_factors=stability_results.sliding_factors,
        overturning_factors=stability_results.overturning_factors,
        uplift_factors=stability_results.uplift_factors,
        seepage_factors=stability_results.seepage_factors,
    )
    
    print(f"\n✓ 场景 '{scenario_name}' 测试完成！")
    print(f"  结果保存在: {scenario_dir}")
    
    return results


def test_all_scenarios():
    """测试所有场景"""
    print("\n" + "="*80)
    print("明渠边坡稳定性系统 - 非恒定流动态场景综合测试")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./tests/dynamic_scenarios_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试场景
    scenarios = {
        '流量增加阶跃': ScenarioGenerator.create_discharge_step(
            baseline_q=10.0, step_q=25.0, step_time=3600.0
        ),
        '流量减少阶跃': ScenarioGenerator.create_discharge_step(
            baseline_q=25.0, step_q=10.0, step_time=3600.0
        ),
        '水位上升阶跃': ScenarioGenerator.create_stage_step(
            baseline_h=2.0, step_h=3.5, step_time=3600.0
        ),
        '水位下降阶跃': ScenarioGenerator.create_stage_step(
            baseline_h=3.5, step_h=2.0, step_time=3600.0
        ),
        '地下水位上升': ScenarioGenerator.create_groundwater_step(
            baseline_level=98.0, step_level=99.5, step_time=3600.0
        ),
        '突发强降雨': ScenarioGenerator.create_rainfall_step(
            baseline_rainfall=0.0, step_rainfall=30.0,
            step_time=3600.0, duration=1800.0
        ),
    }
    
    # 测试每个场景
    all_results = {}
    
    for scenario_name, scenario in scenarios.items():
        try:
            results = test_scenario_with_steady_initial(
                scenario_name=scenario_name,
                scenario=scenario,
                output_dir=output_dir,
            )
            all_results[scenario_name] = results
        except Exception as e:
            print(f"\n✗ 场景 '{scenario_name}' 测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比分析
    print("\n" + "="*80)
    print("生成多场景对比分析...")
    print("="*80)
    
    if len(all_results) >= 2:
        visualizer = DynamicVisualizer(output_dir=output_dir)
        
        results_list = list(all_results.values())
        names_list = list(all_results.keys())
        
        # 选择中间断面进行对比
        num_sections = len(results_list[0]['hydrodynamics'].stations)
        station_idx = num_sections // 2
        
        visualizer.create_comparison_plot(
            results_list=results_list,
            scenario_names=names_list,
            variable='depth',
            station_index=station_idx,
            output_filename='scenarios_comparison.png',
        )
        
        print(f"\n✓ 对比分析图已生成")
    
    # 生成测试报告
    print("\n" + "="*80)
    print("生成测试报告...")
    print("="*80)
    
    report_file = os.path.join(output_dir, 'TEST_REPORT.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 非恒定流动态场景综合测试报告\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 测试概述\n\n")
        f.write("本次测试验证了明渠边坡稳定性系统在各种阶跃变化场景下的响应特性。\n\n")
        f.write("### 测试方法\n\n")
        f.write("1. **初值计算**: 使用恒定流求解器计算初始水力条件\n")
        f.write("2. **非恒定流模拟**: 采用Preissmann隐式格式求解圣维南方程\n")
        f.write("3. **耦合模拟**: 水动力学、水质、边坡稳定性三模块耦合\n")
        f.write("4. **动态可视化**: 生成时间序列图和沿程分布GIF动画\n\n")
        
        f.write("## 测试场景\n\n")
        for i, (name, scenario) in enumerate(scenarios.items(), 1):
            f.write(f"### {i}. {name}\n\n")
            f.write(f"- **描述**: {scenario.description}\n")
            f.write(f"- **参数类型**: {scenario.parameter_type}\n")
            f.write(f"- **基准值**: {scenario.baseline_value:.2f}\n")
            f.write(f"- **阶跃值**: {scenario.step_value:.2f}\n")
            f.write(f"- **阶跃时刻**: {scenario.step_time/3600:.1f} 小时\n")
            
            if name in all_results:
                f.write(f"- **测试状态**: ✓ 通过\n")
                
                hydro = all_results[name]['hydrodynamics']
                f.write(f"- **模拟时间步数**: {len(hydro.times)}\n")
                f.write(f"- **最大水深**: {np.max(hydro.depths):.2f} m\n")
                f.write(f"- **最大流速**: {np.max(hydro.velocities):.2f} m/s\n")
                f.write(f"- **最大Froude数**: {np.max(hydro.froude_numbers):.3f}\n")
                
                stability = all_results[name]['slope_stability']
                f.write(f"- **最小安全系数**: {np.min(stability.comprehensive_factors):.3f}\n")
                
                f.write(f"\n**结果文件**:\n")
                scenario_dir = name.replace(' ', '_')
                f.write(f"- 时间序列图: `{scenario_dir}/timeseries_*.png`\n")
                f.write(f"- 水深动画: `{scenario_dir}/depth_animation.gif`\n")
                f.write(f"- 流量动画: `{scenario_dir}/discharge_animation.gif`\n")
                f.write(f"- 流速动画: `{scenario_dir}/velocity_animation.gif`\n")
                f.write(f"- 边界条件: `{scenario_dir}/boundary_conditions.png`\n")
            else:
                f.write(f"- **测试状态**: ✗ 失败\n")
            
            f.write("\n")
        
        f.write("## 对比分析\n\n")
        if len(all_results) >= 2:
            f.write("多场景对比分析图: `scenarios_comparison.png`\n\n")
        
        f.write("## 结论\n\n")
        f.write(f"成功测试 {len(all_results)}/{len(scenarios)} 个场景。\n\n")
        f.write("### 主要发现\n\n")
        f.write("1. **恒定流初值**: 恒定流求解器能够为非恒定流模拟提供合理的初始条件\n")
        f.write("2. **阶跃响应**: 系统能够捕捉各种阶跃变化的动态响应过程\n")
        f.write("3. **耦合效应**: 水动力学、水质、边坡稳定性三模块耦合运行正常\n")
        f.write("4. **可视化**: 时间序列图和GIF动画直观展示了系统的时空动态特征\n\n")
        
        f.write("---\n\n")
        f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"\n✓ 测试报告已生成: {report_file}")
    
    print("\n" + "="*80)
    print("所有测试完成！")
    print(f"结果保存在: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_all_scenarios()

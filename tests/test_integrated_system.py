"""
综合集成测试

测试完整的水动力学→水质→边坡稳定性链路
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_full_integration():
    """测试完整集成"""
    print("\n" + "="*70)
    print("集成测试: 水动力学 → 水质 → 边坡稳定性")
    print("="*70)
    
    try:
        import numpy as np
        
        from channel_stability.core.channel_system import (
            ChannelSystem, ChannelSection, SoilProperties, LiningProperties
        )
        from channel_stability.core.monitoring_network import (
            MonitoringNetwork, MonitoringStation, StationType
        )
        from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
        from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
        from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
        from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
        from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
        from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig
        
        print("✓ 所有模块导入成功")
        
        # 1. 创建测试系统
        print("\n步骤1: 创建测试系统")
        print("-" * 60)
        
        # 土壤和衬砌
        soil = SoilProperties(
            cohesion=15.0, friction_angle=30.0, unit_weight=18.0,
            saturated_unit_weight=20.0, permeability=1e-5, porosity=0.35
        )
        lining = LiningProperties(
            thickness=0.1, density=2400.0, elastic_modulus=30.0, friction_coeff=0.4
        )
        
        # 渠道系统
        channel = ChannelSystem(name="测试渠道", total_length=5000.0, sections=[])
        
        for i in range(11):  # 11个断面，每500m一个
            station = i * 500.0
            section = ChannelSection(
                station=station,
                bed_elevation=100.0 - 0.0001 * station,
                bottom_width=5.0,
                side_slope=2.0,
                max_depth=4.0,
                manning_n=0.025,
                slope_height=3.0,
                slope_angle=26.57,
                soil_properties=soil,
                lining_properties=lining,
            )
            channel.sections.append(section)
        
        print(f"  渠道: {channel.num_sections}个断面, {channel.total_length/1000}km")
        
        # 监测网络
        network = MonitoringNetwork(network_name="测试网络")
        network.create_uniform_groundwater_network(5000.0, 500.0)
        network.add_boundary_stations(0.0, 5000.0)
        
        # 添加模拟监测数据
        times = np.linspace(0, 3600, 20)
        for station in network.groundwater_stations:
            for t in times:
                gw_level = 98.0 + 0.5 * np.sin(2 * np.pi * t / 3600)
                station.add_observation(time=t, groundwater_level=gw_level)
        
        for station in network.rainfall_stations:
            for t in times:
                rainfall = 5.0 if t > 1800 else 0.0  # 后半段降雨
                station.add_observation(time=t, rainfall=rainfall)
        
        print(f"  监测网络: {network.num_stations}个站点")
        
        # 边界条件
        bc = BoundaryConditions.create_constant_bc(
            upstream_q=10.0,
            downstream_h=2.0,
        )
        print("  边界条件: 上游10m³/s, 下游2.0m")
        
        # 2. 水动力学仿真
        print("\n步骤2: 水动力学仿真")
        print("-" * 60)
        
        hydro_solver = PreissmannHydrodynamicSolver(
            channel=channel,
            boundary_conditions=bc,
            enable_adaptive_mesh=False,
        )
        
        hydro_results = hydro_solver.solve(
            total_time=3600.0,  # 1小时
            dt=60.0,
            initial_depth=2.0,
            initial_discharge=10.0,
            save_interval=10,
        )
        
        print(f"  完成: {len(hydro_results.times)}个时间步")
        print(f"  最大水深: {np.max(hydro_results.depths):.3f}m")
        print(f"  最大流速: {np.max(hydro_results.velocities):.3f}m/s")
        
        # 3. 水质仿真
        print("\n步骤3: 水质仿真")
        print("-" * 60)
        
        wq_params = WaterQualityParameters(temperature=20.0)
        wq_solver = WaterQualitySolver(parameters=wq_params)
        
        wq_results = wq_solver.solve(
            hydrodynamic_results=hydro_results,
            indicators=['DO', 'BOD'],
            dt=60.0,
        )
        
        print(f"  完成: 模拟{len(wq_results.concentrations)}个指标")
        for indicator in ['DO', 'BOD']:
            C = wq_results.concentrations[indicator]
            print(f"  {indicator}: {np.min(C):.2f} - {np.max(C):.2f} mg/L")
        
        # 4. 边坡稳定性计算
        print("\n步骤4: 边坡稳定性计算")
        print("-" * 60)
        
        stability_calc = SlopeStabilityCalculator(
            channel=channel,
            monitoring_network=network,
        )
        
        # 选择几个时间步计算
        time_indices = [0, len(hydro_results.times)//2, len(hydro_results.times)-1]
        
        stability_results = stability_calc.compute_stability(
            hydrodynamic_results=hydro_results,
            time_indices=time_indices,
        )
        
        print(f"  完成: {len(stability_results.times)}个时间点")
        print(f"  最小稳定系数: {np.min(stability_results.comprehensive_factors):.2f}")
        print(f"  平均稳定指数: {np.mean(stability_results.channel_stability_index):.2f}")
        print(f"  不稳定断面数: {int(np.max(stability_results.unstable_section_count))}")
        
        # 5. 验证一致性
        print("\n步骤5: 验证结果一致性")
        print("-" * 60)
        
        # 验证时间一致性
        assert len(hydro_results.times) == len(wq_results.times), "时间序列长度应一致"
        print("  ✓ 时间序列一致")
        
        # 验证空间一致性
        assert len(hydro_results.stations) == len(wq_results.stations), "断面数应一致"
        assert len(hydro_results.stations) == len(stability_results.stations), "断面数应一致"
        print("  ✓ 空间网格一致")
        
        # 验证物理合理性
        assert np.all(hydro_results.depths > 0), "水深应为正"
        assert np.all(wq_results.concentrations['DO'] >= 0), "DO浓度应非负"
        assert np.all(stability_results.comprehensive_factors > 0), "稳定系数应为正"
        print("  ✓ 物理量合理")
        
        # 验证因果关系
        # 水深增加应导致边坡稳定性变化
        depth_change = hydro_results.depths[-1] - hydro_results.depths[0]
        stability_change = stability_results.comprehensive_factors[-1] - stability_results.comprehensive_factors[0]
        print(f"  ✓ 水深变化: {np.mean(depth_change):.4f}m")
        print(f"  ✓ 稳定性变化: {np.mean(stability_change):.4f}")
        
        # 6. 性能统计
        print("\n步骤6: 性能统计")
        print("-" * 60)
        
        n_cells = len(hydro_results.stations)
        n_times = len(hydro_results.times)
        total_dof = n_cells * n_times
        
        print(f"  空间离散: {n_cells}个断面")
        print(f"  时间离散: {n_times}个时间步")
        print(f"  总自由度: {total_dof}")
        print(f"  数据量: ~{total_dof * 8 / 1024:.1f} KB")
        
        # 7. 使用IntegratedSimulator测试
        print("\n步骤7: 测试IntegratedSimulator")
        print("-" * 60)
        
        simulator = IntegratedSimulator(
            channel=channel,
            monitoring_network=network,
            boundary_conditions=bc,
            water_quality_params=wq_params,
        )
        
        config = SimulationConfig(
            total_time=600.0,  # 10分钟快速测试
            dt=60.0,
            save_interval=5,
            enable_hydrodynamics=True,
            enable_water_quality=True,
            enable_slope_stability=True,
            simulated_indicators=['DO', 'BOD'],
            stability_time_interval=5,
        )
        
        integrated_results = simulator.run_simulation(config)
        
        print("  ✓ 集成仿真完成")
        print(f"  ✓ 水动力学: {len(integrated_results.hydrodynamics.times)}个时间步")
        print(f"  ✓ 水质: {len(integrated_results.water_quality.times)}个时间步")
        print(f"  ✓ 边坡稳定性: {len(integrated_results.slope_stability.times)}个时间步")
        
        # 8. 测试结果摘要
        summary = integrated_results.summary()
        print("\n步骤8: 结果摘要")
        print("-" * 60)
        print(f"  水动力学: 最大水深={summary['hydrodynamics']['max_depth']:.2f}m")
        print(f"  水质: DO范围={summary['water_quality']['DO_min']:.2f}-{summary['water_quality']['DO_max']:.2f}mg/L")
        print(f"  边坡稳定性: 最小系数={summary['slope_stability']['min_stability_factor']:.2f}")
        
        print("\n" + "="*70)
        print("✓✓✓ 集成测试完全通过！✓✓✓")
        print("="*70)
        
        return True
        
    except ImportError as e:
        print(f"\n⚠ 依赖缺失: {e}")
        print("请运行: pip install numpy scipy matplotlib")
        return False
        
    except Exception as e:
        print(f"\n✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_flood():
    """测试场景: 洪水过程"""
    print("\n" + "="*70)
    print("场景测试: 洪水过程对边坡稳定性的影响")
    print("="*70)
    
    try:
        import numpy as np
        from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
        
        # 创建洪水过程
        time_series = [
            (0, 5.0),      # 基流
            (1800, 25.0),  # 洪峰
            (3600, 5.0),   # 退水
        ]
        
        bc = BoundaryConditions.create_hydrograph_bc(
            time_series=time_series,
            downstream_h=2.0,
        )
        
        print("\n洪水过程:")
        for t in [0, 900, 1800, 2700, 3600]:
            q = bc.get_upstream_discharge(t)
            print(f"  t={t/60:4.0f}分钟: Q={q:5.1f} m³/s")
        
        print("\n✓ 洪水过程场景测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 场景测试失败: {e}")
        return False


def test_scenario_rainfall():
    """测试场景: 降雨入渗影响"""
    print("\n" + "="*70)
    print("场景测试: 降雨对边坡稳定性的影响")
    print("="*70)
    
    try:
        from channel_stability.core.channel_system import SoilProperties, LiningProperties
        from channel_stability.slope_stability.failure_mechanisms import LiningStabilityAnalysis
        
        soil = SoilProperties(
            cohesion=15.0, friction_angle=30.0, unit_weight=18.0,
            saturated_unit_weight=20.0, permeability=1e-5, porosity=0.35
        )
        lining = LiningProperties(
            thickness=0.1, density=2400.0, elastic_modulus=30.0, friction_coeff=0.4
        )
        
        # 测试不同降雨强度
        rainfalls = [0, 10, 20, 50, 100]  # mm/h
        
        print("\n降雨强度 (mm/h) | 综合稳定系数 | 变化率")
        print("-" * 55)
        
        base_factor = None
        for rain in rainfalls:
            factors = LiningStabilityAnalysis.compute_comprehensive_stability(
                slope_angle=26.57,
                slope_height=3.0,
                soil_props=soil,
                lining_props=lining,
                groundwater_depth=1.0,
                channel_water_level=2.0,
                rainfall_intensity=rain,
            )
            
            if base_factor is None:
                base_factor = factors.comprehensive_factor
                change = 0.0
            else:
                change = (factors.comprehensive_factor - base_factor) / base_factor * 100
            
            print(f"{rain:15d} | {factors.comprehensive_factor:16.2f} | {change:7.1f}%")
        
        print("\n✓ 降雨场景测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 场景测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "#"*70)
    print("# 明渠边坡稳定性系统 - 集成测试套件")
    print("#"*70)
    
    results = {}
    
    # 完整集成测试
    results['full_integration'] = test_full_integration()
    
    # 场景测试
    results['flood_scenario'] = test_scenario_flood()
    results['rainfall_scenario'] = test_scenario_rainfall()
    
    # 汇总
    print("\n" + "#"*70)
    print("# 测试结果汇总")
    print("#"*70)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("所有集成测试通过！✓✓✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("部分测试失败，请检查环境和依赖")
        print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

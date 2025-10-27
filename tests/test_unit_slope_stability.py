"""
边坡稳定性模块最小单元测试

测试目标：
1. 验证失稳机理计算
2. 验证稳定性计算器
3. 验证综合稳定性评估
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_failure_mechanisms():
    """测试失稳机理"""
    print("\n" + "="*60)
    print("测试1: 失稳机理计算")
    print("="*60)
    
    from channel_stability.core.channel_system import SoilProperties, LiningProperties
    from channel_stability.slope_stability.failure_mechanisms import LiningStabilityAnalysis
    
    # 创建土壤特性
    soil = SoilProperties(
        cohesion=15.0,  # kPa
        friction_angle=30.0,  # 度
        unit_weight=18.0,  # kN/m³
        saturated_unit_weight=20.0,  # kN/m³
        permeability=1e-5,  # m/s
        porosity=0.35,
    )
    
    # 创建衬砌特性
    lining = LiningProperties(
        thickness=0.1,  # m
        density=2400.0,  # kg/m³
        elastic_modulus=30.0,  # GPa
        friction_coeff=0.4,
    )
    
    # 边坡参数
    slope_angle = 26.57  # 度，对应1:2边坡
    slope_height = 3.0  # m
    
    print(f"\n边坡参数:")
    print(f"  坡角: {slope_angle}°")
    print(f"  坡高: {slope_height} m")
    print(f"\n土壤参数:")
    print(f"  粘聚力: {soil.cohesion} kPa")
    print(f"  内摩擦角: {soil.friction_angle}°")
    print(f"  容重: {soil.unit_weight} kN/m³")
    
    # 测试不同地下水位情况
    print("\n" + "-"*60)
    print("测试不同地下水位下的稳定性")
    print("-"*60)
    
    groundwater_depths = [0.0, 1.0, 2.0, 3.0]  # 地下水埋深
    channel_water_level = 2.0  # 渠道水位
    rainfall = 0.0  # 无降雨
    
    print(f"\n渠道水位: {channel_water_level} m")
    print(f"降雨强度: {rainfall} mm/h")
    print("\n地下水埋深(m) | 滑动系数 | 倾覆系数 | 浮托系数 | 渗透系数 | 综合系数 | 是否稳定")
    print("-" * 95)
    
    for gw_depth in groundwater_depths:
        factors = LiningStabilityAnalysis.compute_comprehensive_stability(
            slope_angle=slope_angle,
            slope_height=slope_height,
            soil_props=soil,
            lining_props=lining,
            groundwater_depth=gw_depth,
            channel_water_level=channel_water_level,
            rainfall_intensity=rainfall,
        )
        
        print(f"{gw_depth:15.1f} | {factors.sliding_factor:10.2f} | "
              f"{factors.overturning_factor:10.2f} | {factors.uplift_factor:10.2f} | "
              f"{factors.seepage_factor:10.2f} | {factors.comprehensive_factor:10.2f} | "
              f"{'是' if factors.is_stable else '否':^8s}")
        
        # 验证
        assert factors.sliding_factor > 0, "滑动系数应为正"
        assert factors.overturning_factor > 0, "倾覆系数应为正"
        assert factors.uplift_factor > 0, "浮托系数应为正"
        assert factors.seepage_factor > 0, "渗透系数应为正"
        assert factors.comprehensive_factor > 0, "综合系数应为正"
    
    # 测试降雨影响
    print("\n" + "-"*60)
    print("测试降雨对稳定性的影响")
    print("-"*60)
    
    gw_depth = 1.0  # 固定地下水埋深
    rainfalls = [0, 5, 10, 20, 50]  # mm/h
    
    print(f"\n地下水埋深: {gw_depth} m")
    print(f"渠道水位: {channel_water_level} m")
    print("\n降雨强度(mm/h) | 滑动系数 | 综合系数 | 是否稳定")
    print("-" * 55)
    
    for rain in rainfalls:
        factors = LiningStabilityAnalysis.compute_comprehensive_stability(
            slope_angle=slope_angle,
            slope_height=slope_height,
            soil_props=soil,
            lining_props=lining,
            groundwater_depth=gw_depth,
            channel_water_level=channel_water_level,
            rainfall_intensity=rain,
        )
        
        print(f"{rain:16d} | {factors.sliding_factor:10.2f} | "
              f"{factors.comprehensive_factor:10.2f} | "
              f"{'是' if factors.is_stable else '否':^8s}")
    
    print("\n✓ 失稳机理测试通过")
    return soil, lining


def test_stability_calculator():
    """测试稳定性计算器"""
    print("\n" + "="*60)
    print("测试2: 稳定性计算器")
    print("="*60)
    
    from channel_stability.core.channel_system import (
        ChannelSystem, ChannelSection, SoilProperties, LiningProperties
    )
    from channel_stability.core.monitoring_network import (
        MonitoringNetwork, MonitoringStation, StationType
    )
    from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
    from channel_stability.hydrodynamics.preissmann_solver import HydrodynamicResults
    
    # 创建简单渠道系统（3个断面）
    soil = SoilProperties(
        cohesion=15.0, friction_angle=30.0, unit_weight=18.0,
        saturated_unit_weight=20.0, permeability=1e-5, porosity=0.35
    )
    
    lining = LiningProperties(
        thickness=0.1, density=2400.0, elastic_modulus=30.0, friction_coeff=0.4
    )
    
    channel = ChannelSystem(
        name="测试渠道",
        total_length=1000.0,
        sections=[],
    )
    
    for i in range(3):
        station = i * 500.0
        section = ChannelSection(
            station=station,
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
        channel.sections.append(section)
    
    print(f"\n渠道系统:")
    print(f"  断面数: {channel.num_sections}")
    print(f"  总长度: {channel.total_length} m")
    
    # 创建监测网络
    network = MonitoringNetwork(network_name="测试网络")
    
    # 添加地下水观测井
    for i, section in enumerate(channel.sections):
        station = MonitoringStation(
            station_id=f"GW_{i}",
            station_type=StationType.GROUNDWATER,
            station=section.station,
        )
        # 添加模拟数据
        times = [0, 300, 600]
        gw_levels = [98.0, 98.5, 99.0]  # 地下水位上升
        for t, gw in zip(times, gw_levels):
            station.add_observation(time=t, groundwater_level=gw)
        
        network.stations.append(station)
    
    # 添加雨量站
    for loc in [0, 1000]:
        station = MonitoringStation(
            station_id=f"RAIN_{loc}",
            station_type=StationType.RAINFALL,
            station=loc,
        )
        for t in [0, 300, 600]:
            station.add_observation(time=t, rainfall=10.0)
        network.stations.append(station)
    
    print(f"\n监测网络:")
    print(f"  总站点数: {network.num_stations}")
    print(f"  地下水站: {len(network.groundwater_stations)}")
    print(f"  雨量站: {len(network.rainfall_stations)}")
    
    # 创建模拟水动力学结果
    times = np.array([0, 300, 600])
    stations = channel.stations
    depths = np.array([
        [2.0, 2.0, 2.0],  # t=0
        [2.2, 2.2, 2.2],  # t=300
        [2.5, 2.5, 2.5],  # t=600
    ])
    discharges = np.full((3, 3), 10.0)
    velocities = np.full((3, 3), 0.5)
    water_levels = 100.0 + depths
    froude_numbers = np.full((3, 3), 0.2)
    
    hydro_results = HydrodynamicResults(
        times=times,
        stations=stations,
        depths=depths,
        discharges=discharges,
        velocities=velocities,
        water_levels=water_levels,
        froude_numbers=froude_numbers,
    )
    
    print(f"\n水动力学结果:")
    print(f"  时间点: {len(times)}")
    print(f"  断面数: {len(stations)}")
    
    # 稳定性计算
    print("\n运行稳定性计算...")
    calculator = SlopeStabilityCalculator(
        channel=channel,
        monitoring_network=network,
    )
    
    stability_results = calculator.compute_stability(
        hydrodynamic_results=hydro_results,
    )
    
    print(f"\n稳定性结果:")
    print(f"  计算时间点: {len(stability_results.times)}")
    print(f"  断面数: {len(stability_results.stations)}")
    
    # 显示结果
    print("\n各断面稳定性系数:")
    print("时间(s) | 断面   | 滑动 | 倾覆 | 浮托 | 渗透 | 综合 | 状态")
    print("-" * 70)
    
    for t_idx, t in enumerate(stability_results.times):
        for s_idx, s in enumerate(stability_results.stations):
            print(f"{t:7.0f} | {s:6.0f} | "
                  f"{stability_results.sliding_factors[t_idx, s_idx]:4.2f} | "
                  f"{stability_results.overturning_factors[t_idx, s_idx]:4.2f} | "
                  f"{stability_results.uplift_factors[t_idx, s_idx]:4.2f} | "
                  f"{stability_results.seepage_factors[t_idx, s_idx]:4.2f} | "
                  f"{stability_results.comprehensive_factors[t_idx, s_idx]:4.2f} | "
                  f"{'稳定' if stability_results.stability_status[t_idx, s_idx] else '不稳定':^6s}")
    
    # 全渠道指标
    print(f"\n全渠道稳定性指标:")
    for t_idx, t in enumerate(stability_results.times):
        print(f"  t={t:4.0f}s: 综合指数={stability_results.channel_stability_index[t_idx]:.2f}, "
              f"不稳定断面数={stability_results.unstable_section_count[t_idx]:.0f}")
    
    # 验证
    assert np.all(stability_results.comprehensive_factors > 0), "综合系数应为正"
    assert np.all(np.isfinite(stability_results.comprehensive_factors)), "系数应为有限值"
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 各失稳模式系数分布（最后时刻）
    axes[0, 0].plot(stations, stability_results.sliding_factors[-1], 'o-', 
                    label='滑动', linewidth=2)
    axes[0, 0].plot(stations, stability_results.overturning_factors[-1], 's-', 
                    label='倾覆', linewidth=2)
    axes[0, 0].plot(stations, stability_results.uplift_factors[-1], '^-', 
                    label='浮托', linewidth=2)
    axes[0, 0].axhline(y=1.3, color='r', linestyle='--', alpha=0.5, label='阈值')
    axes[0, 0].set_xlabel('桩号 (m)')
    axes[0, 0].set_ylabel('稳定系数')
    axes[0, 0].set_title('各失稳模式稳定系数')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 综合稳定系数分布
    for t_idx, t in enumerate(stability_results.times):
        axes[0, 1].plot(stations, stability_results.comprehensive_factors[t_idx], 
                       'o-', label=f't={t:.0f}s', linewidth=2)
    axes[0, 1].axhline(y=1.3, color='r', linestyle='--', label='安全阈值')
    axes[0, 1].set_xlabel('桩号 (m)')
    axes[0, 1].set_ylabel('综合稳定系数')
    axes[0, 1].set_title('综合稳定系数时空分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 全渠道稳定性指数时间历程
    axes[1, 0].plot(stability_results.times, stability_results.channel_stability_index, 
                   'b-o', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=1.3, color='r', linestyle='--', label='安全阈值')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('全渠道稳定性指数')
    axes[1, 0].set_title('全渠道稳定性时间历程')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 不稳定断面数量
    axes[1, 1].bar(range(len(stability_results.times)), 
                  stability_results.unstable_section_count,
                  color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('不稳定断面数量')
    axes[1, 1].set_title('不稳定断面统计')
    axes[1, 1].set_xticks(range(len(stability_results.times)))
    axes[1, 1].set_xticklabels([f'{t:.0f}s' for t in stability_results.times])
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_slope_stability_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果已保存: test_slope_stability_results.png")
    plt.close()
    
    print("\n✓ 稳定性计算器测试通过")
    return stability_results


def run_all_slope_stability_tests():
    """运行所有边坡稳定性测试"""
    print("\n" + "#"*60)
    print("# 边坡稳定性模块单元测试")
    print("#"*60)
    
    try:
        # 测试1: 失稳机理
        soil, lining = test_failure_mechanisms()
        
        # 测试2: 稳定性计算器
        stability_results = test_stability_calculator()
        
        print("\n" + "#"*60)
        print("# 边坡稳定性模块测试 - 全部通过 ✓")
        print("#"*60)
        return True, stability_results
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, results = run_all_slope_stability_tests()
    exit(0 if success else 1)

"""
边坡稳定性监测预测系统示例（使用修复后的求解器）

展示完整的工作流程，使用准稳态求解器（完全守恒）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from channel_stability.core.channel_system import (
    ChannelSystem, ChannelSection, SoilProperties, LiningProperties
)
from channel_stability.core.monitoring_network import MonitoringNetwork
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig


def main():
    """完整示例"""
    print("\n" + "="*70)
    print("边坡稳定性监测预测系统完整示例")
    print("="*70 + "\n")
    
    # 1. 创建明渠系统
    print("1. 创建明渠系统...")
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
    
    channel = ChannelSystem(name="示例明渠", total_length=5000.0, sections=[])
    
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
        num_sections=11,
        base_section=base_section,
        bed_slope=0.0002,
    )
    
    print(f"  创建明渠: 长度{channel.total_length}m, {channel.num_sections}个断面")
    
    # 2. 创建监测网络
    print("\n2. 创建监测网络...")
    network = MonitoringNetwork(network_name="示例监测网络")
    network.create_uniform_groundwater_network(
        channel_length=channel.total_length,
        spacing=1000.0,
    )
    network.add_boundary_stations(0.0, channel.total_length)
    
    # 初始化监测数据
    times = np.linspace(0, 7200, 20)
    for station in network.groundwater_stations:
        for t in times:
            station.add_observation(time=t, groundwater_level=98.0)
    for station in network.rainfall_stations:
        for t in times:
            station.add_observation(time=t, rainfall=0.0)
    
    print(f"  创建监测网络: {network.num_stations}个站点")
    
    # 3. 设置边界条件（流量从10增加到25 m³/s）
    print("\n3. 设置边界条件...")
    
    def upstream_q(t):
        if t < 3600:  # 前1小时
            return 10.0
        else:  # 后1小时
            return 25.0
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=upstream_q,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    print("  上游边界: 流量 10→25 m³/s (1小时后阶跃)")
    print("  下游边界: 水位 102.5 m (固定)")
    
    # 4. 创建综合仿真器
    print("\n4. 创建综合仿真器...")
    simulator = IntegratedSimulator(
        channel=channel,
        monitoring_network=network,
        boundary_conditions=bc,
    )
    
    # 5. 运行仿真（使用准稳态求解器）
    print("\n5. 运行仿真...")
    config = SimulationConfig(
        total_time=7200.0,  # 2小时
        dt=60.0,  # 1分钟时间步
        save_interval=1,
        hydrodynamic_solver_type="quasi_steady",  # 使用准稳态求解器 ⭐
        enable_hydrodynamics=True,
        enable_water_quality=True,
        enable_slope_stability=True,
        simulated_indicators=['DO', 'BOD', 'NH3N'],
    )
    
    results = simulator.run_simulation(config)
    
    # 6. 分析结果
    print("\n6. 结果分析...")
    
    # 水动力学
    hydro = results.hydrodynamics
    if hydro is not None:
        v_max = np.max(hydro.velocities)
        v_min = np.min(hydro.velocities)
        fr_max = np.max(hydro.froude_numbers)
        
        q_in = hydro.discharges[:, 0]
        q_out = hydro.discharges[:, -1]
        mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
        
        print(f"\n水动力学结果:")
        print(f"  流速范围: [{v_min:.3f}, {v_max:.3f}] m/s")
        print(f"  Froude数: [0, {fr_max:.3f}]")
        print(f"  质量守恒误差: {mass_error*100:.4f}%")
    
    # 边坡稳定性
    stability = results.slope_stability
    if stability is not None:
        fs_min = np.min(stability.comprehensive_factors)
        fs_max = np.max(stability.comprehensive_factors)
        unstable_count = np.max(stability.unstable_section_count)
        
        print(f"\n边坡稳定性结果:")
        print(f"  安全系数范围: [{fs_min:.3f}, {fs_max:.3f}]")
        print(f"  最大不稳定断面数: {int(unstable_count)}")
        print(f"  全渠道稳定性指数: {np.mean(stability.channel_stability_index):.3f}")
    
    # 7. 生成图表
    print("\n7. 生成结果图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if hydro is not None:
        times_h = hydro.times / 3600.0
        mid_idx = len(hydro.stations) // 2
        
        # 水深
        axes[0, 0].plot(times_h, hydro.depths[:, mid_idx], 'b-', linewidth=2)
        axes[0, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Step time')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Depth (m)')
        axes[0, 0].set_title('Depth Evolution (Middle Station)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 流速
        axes[0, 1].plot(times_h, hydro.velocities[:, mid_idx], 'g-', linewidth=2)
        axes[0, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=5.0, color='k', linestyle=':', alpha=0.3)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 质量守恒
        axes[1, 0].plot(times_h, q_in, 'b-', linewidth=2, label='Inlet')
        axes[1, 0].plot(times_h, q_out, 'r--', linewidth=2, label='Outlet')
        axes[1, 0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Discharge (m³/s)')
        axes[1, 0].set_title('Mass Conservation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 安全系数
    if stability is not None:
        stab_times_h = stability.times / 3600.0
        axes[1, 1].plot(stab_times_h, stability.channel_stability_index, 
                       'r-', linewidth=2, marker='o', markersize=4)
        axes[1, 1].axhline(y=1.3, color='orange', linestyle='--', label='Safety threshold')
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Critical')
        axes[1, 1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Stability Index')
        axes[1, 1].set_title('Channel Stability Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = './examples/channel_stability_example_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {output_file}")
    
    # 8. 总结
    print("\n" + "="*70)
    print("✓ 示例运行成功！")
    print("="*70)
    print("\n关键指标:")
    if hydro is not None:
        print(f"  ✓ 质量守恒误差: {mass_error*100:.4f}% (目标<5%)")
        print(f"  ✓ 流速范围: [{v_min:.3f}, {v_max:.3f}] m/s")
        print(f"  ✓ Froude数: [0, {fr_max:.3f}]")
    if stability is not None:
        print(f"  ✓ 最小安全系数: {fs_min:.3f}")
    
    print("\n使用的求解器: 准稳态求解器（完全守恒）")
    print("推荐: 边坡稳定性评估优先使用此求解器\n")


if __name__ == "__main__":
    main()

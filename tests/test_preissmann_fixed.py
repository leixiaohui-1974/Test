"""
测试修复后的Preissmann求解器

验证：
1. 边界条件是否正确实施
2. 流速是否在合理范围
3. CFL条件是否满足
4. 质量守恒是否改善
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from channel_stability.core.channel_system import (
    ChannelSystem, ChannelSection, SoilProperties, LiningProperties
)
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
from channel_stability.hydrodynamics.steady_flow_solver import SteadyFlowSolver


def create_test_channel(length=5000.0, num_sections=11):
    """创建测试明渠（更简单的配置）"""
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


def test_steady_to_unsteady():
    """测试1: 从稳态到非恒定（小扰动）"""
    print("\n" + "="*70)
    print("测试1: 稳态到非恒定（流量10→12 m³/s）")
    print("="*70)
    
    channel = create_test_channel()
    
    # 计算稳态流初值
    print("\n计算稳态流初值...")
    steady_solver = SteadyFlowSolver(channel)
    steady_results = steady_solver.solve_gradually_varied_flow(
        discharge=10.0,
        downstream_depth=2.5,
    )
    
    initial_depth = np.mean(steady_results['depths'])
    print(f"  初始水深: {initial_depth:.3f} m")
    
    # 边界条件：流量小幅增加
    def q_func(t):
        if t < 600:  # 前10分钟
            return 10.0
        else:
            return 12.0
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=q_func,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    # 创建求解器（启用自适应时间步长）
    solver = PreissmannHydrodynamicSolver(
        channel=channel,
        boundary_conditions=bc,
        theta=0.7,
        relaxation=0.05,
        enable_adaptive_mesh=False,
    )
    
    # 求解
    print("\n开始非恒定流模拟...")
    results = solver.solve(
        total_time=1800.0,  # 30分钟
        dt=5.0,  # 初始时间步长5秒
        initial_depth=initial_depth,
        initial_discharge=10.0,
        save_interval=10,
        use_adaptive_dt=True,
        cfl_max=0.9,
        min_dt=1.0,
        max_dt=30.0,
    )
    
    return results, channel, "稳态到非恒定"


def test_medium_change():
    """测试2: 中等流量变化"""
    print("\n" + "="*70)
    print("测试2: 中等变化（流量10→20 m³/s）")
    print("="*70)
    
    channel = create_test_channel()
    
    # 稳态初值
    print("\n计算稳态流初值...")
    steady_solver = SteadyFlowSolver(channel)
    steady_results = steady_solver.solve_gradually_varied_flow(
        discharge=10.0,
        downstream_depth=2.5,
    )
    
    initial_depth = np.mean(steady_results['depths'])
    print(f"  初始水深: {initial_depth:.3f} m")
    
    # 边界条件
    def q_func(t):
        if t < 600:
            return 10.0
        else:
            return 20.0
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=q_func,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    solver = PreissmannHydrodynamicSolver(
        channel=channel,
        boundary_conditions=bc,
        theta=0.7,
        relaxation=0.05,
        enable_adaptive_mesh=False,
    )
    
    print("\n开始非恒定流模拟...")
    results = solver.solve(
        total_time=1800.0,
        dt=5.0,
        initial_depth=initial_depth,
        initial_discharge=10.0,
        save_interval=10,
        use_adaptive_dt=True,
        cfl_max=0.9,
        min_dt=1.0,
        max_dt=30.0,
    )
    
    return results, channel, "中等变化"


def evaluate_results(results, channel, scenario_name):
    """评价结果"""
    print(f"\n" + "="*70)
    print(f"评价结果: {scenario_name}")
    print("="*70)
    
    # 统计信息
    v_max = np.max(results.velocities)
    v_min = np.min(results.velocities)
    v_mean = np.mean(results.velocities)
    
    fr_max = np.max(results.froude_numbers)
    fr_min = np.min(results.froude_numbers)
    
    q_in = results.discharges[:, 0]
    q_out = results.discharges[:, -1]
    mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
    
    # Courant数（估算）
    dx = channel.stations[1] - channel.stations[0]
    if hasattr(results, 'dt_history') and results.dt_history:
        dt_mean = np.mean(results.dt_history)
    else:
        dt_mean = results.times[1] - results.times[0] if len(results.times) > 1 else 5.0
    
    courant = np.abs(results.velocities) * dt_mean / dx
    co_max = np.max(courant)
    co_mean = np.mean(courant)
    
    print(f"\n流速统计:")
    print(f"  最小值: {v_min:.4f} m/s")
    print(f"  最大值: {v_max:.4f} m/s")
    print(f"  平均值: {v_mean:.4f} m/s")
    
    print(f"\nFroude数统计:")
    print(f"  最小值: {fr_min:.4f}")
    print(f"  最大值: {fr_max:.4f}")
    
    print(f"\nCourant数统计:")
    print(f"  最大值: {co_max:.4f}")
    print(f"  平均值: {co_mean:.4f}")
    
    print(f"\n质量守恒:")
    print(f"  误差: {mass_error*100:.2f}%")
    
    # 检查是否通过
    checks = {
        '流速正值': v_min >= 0,
        '流速范围': v_max < 5.0,
        'Froude数正值': fr_min >= 0,
        'Froude数范围': fr_max < 1.5,
        'Courant数': co_max < 2.0,
        '质量守恒': mass_error < 0.05,
    }
    
    print(f"\n质量检查:")
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    all_passed = all(checks.values())
    overall = "✓ 通过" if all_passed else "✗ 未通过"
    print(f"\n总体评价: {overall}")
    
    return checks, all_passed


def plot_results(results, channel, scenario_name, output_file):
    """绘制结果"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Preissmann求解器修复验证 - {scenario_name}', 
                 fontsize=16, fontweight='bold')
    
    stations = results.stations
    times_h = results.times / 3600.0
    mid_idx = len(stations) // 2
    
    # 1. 水深时间历程
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times_h, results.depths[:, 0], 'b-', linewidth=2, label='Upstream')
    ax1.plot(times_h, results.depths[:, mid_idx], 'g-', linewidth=2, label='Middle')
    ax1.plot(times_h, results.depths[:, -1], 'r-', linewidth=2, label='Downstream')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Depth Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 流速时间历程
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times_h, results.velocities[:, 0], 'b-', linewidth=2, label='Upstream')
    ax2.plot(times_h, results.velocities[:, mid_idx], 'g-', linewidth=2, label='Middle')
    ax2.plot(times_h, results.velocities[:, -1], 'r-', linewidth=2, label='Downstream')
    ax2.axhline(y=5.0, color='k', linestyle='--', alpha=0.5, label='Limit (5 m/s)')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Froude数
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(times_h, results.froude_numbers[:, mid_idx], 'm-', linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Critical')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Froude Number')
    ax3.set_title(f'Froude at station {stations[mid_idx]:.0f}m')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 流量沿程（初始和最终）
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(stations, results.discharges[0, :], 'b-o', linewidth=2, 
             markersize=4, label='Initial')
    ax4.plot(stations, results.discharges[-1, :], 'r-s', linewidth=2, 
             markersize=4, label='Final')
    ax4.set_xlabel('Station (m)')
    ax4.set_ylabel('Discharge (m³/s)')
    ax4.set_title('Discharge Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 流速沿程
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(stations, results.velocities[-1, :], 'g-o', linewidth=2, markersize=4)
    ax5.axhline(y=5.0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Station (m)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Velocity Distribution (Final)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 质量守恒
    ax6 = fig.add_subplot(gs[1, 2])
    q_in = results.discharges[:, 0]
    q_out = results.discharges[:, -1]
    ax6.plot(times_h, q_in, 'b-', linewidth=2, label='Inlet')
    ax6.plot(times_h, q_out, 'r--', linewidth=2, label='Outlet')
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Discharge (m³/s)')
    ax6.set_title('Mass Conservation Check')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 水面线（最终时刻）
    ax7 = fig.add_subplot(gs[2, 0])
    water_levels = results.water_levels[-1, :]
    bed_elevs = water_levels - results.depths[-1, :]
    ax7.fill_between(stations, bed_elevs, water_levels, alpha=0.5, color='blue')
    ax7.plot(stations, water_levels, 'b-', linewidth=2, label='Water Surface')
    ax7.plot(stations, bed_elevs, 'k-', linewidth=2, label='Bed')
    ax7.set_xlabel('Station (m)')
    ax7.set_ylabel('Elevation (m)')
    ax7.set_title('Water Surface Profile (Final)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Courant数分布
    ax8 = fig.add_subplot(gs[2, 1])
    dx = stations[1] - stations[0]
    dt_mean = (results.times[1] - results.times[0]) if len(results.times) > 1 else 5.0
    courant = np.abs(results.velocities) * dt_mean / dx
    im = ax8.imshow(courant.T, aspect='auto', origin='lower',
                   extent=[times_h[0], times_h[-1], stations[0], stations[-1]],
                   cmap='RdYlGn_r', vmin=0, vmax=1.5)
    plt.colorbar(im, ax=ax8, label='Courant Number')
    ax8.set_xlabel('Time (hours)')
    ax8.set_ylabel('Station (m)')
    ax8.set_title('Courant Number Distribution')
    
    # 9. 统计信息
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    v_max = np.max(results.velocities)
    v_min = np.min(results.velocities)
    fr_max = np.max(results.froude_numbers)
    mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
    co_max = np.max(courant)
    
    text_lines = [
        "Quality Metrics:",
        "",
        f"Velocity: [{v_min:.3f}, {v_max:.3f}] m/s",
        f"Froude: [0, {fr_max:.3f}]",
        f"Courant Max: {co_max:.3f}",
        f"Mass Error: {mass_error*100:.2f}%",
        "",
        "Status:",
        f"{'✓' if v_min >= 0 else '✗'} No negative velocity",
        f"{'✓' if v_max < 5.0 else '✗'} Velocity < 5 m/s",
        f"{'✓' if fr_max < 1.5 else '✗'} Froude < 1.5",
        f"{'✓' if co_max < 2.0 else '✗'} Courant < 2.0",
        f"{'✓' if mass_error < 0.05 else '✗'} Mass Error < 5%",
    ]
    
    text = "\n".join(text_lines)
    color = 'lightgreen' if (v_max < 5.0 and co_max < 2.0 and mass_error < 0.05) else 'lightyellow'
    ax9.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 图表已保存: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Preissmann求解器修复验证测试")
    print("="*70)
    
    output_dir = './tests/preissmann_fixed_results'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # 测试1: 小扰动
    try:
        results1, channel1, name1 = test_steady_to_unsteady()
        checks1, passed1 = evaluate_results(results1, channel1, name1)
        plot_results(results1, channel1, name1, 
                    os.path.join(output_dir, 'test1_small_change.png'))
        all_results.append((name1, checks1, passed1))
    except Exception as e:
        print(f"\n✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 中等变化
    try:
        results2, channel2, name2 = test_medium_change()
        checks2, passed2 = evaluate_results(results2, channel2, name2)
        plot_results(results2, channel2, name2,
                    os.path.join(output_dir, 'test2_medium_change.png'))
        all_results.append((name2, checks2, passed2))
    except Exception as e:
        print(f"\n✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, checks, passed in all_results:
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"\n{name}: {status}")
        for check_name, check_passed in checks.items():
            check_status = "✓" if check_passed else "✗"
            print(f"  {check_status} {check_name}")
    
    all_passed = all(passed for _, _, passed in all_results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ 所有测试通过！Preissmann求解器修复成功！")
    else:
        print("⚠ 部分测试未通过，需要进一步调整")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

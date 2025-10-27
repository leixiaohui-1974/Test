"""
测试修复后的Preissmann求解器

验证：
1. 自适应时间步长是否工作
2. 流速是否在合理范围
3. Courant数是否满足CFL条件
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
from channel_stability.hydrodynamics.preissmann_solver_fixed import PreissmannHydrodynamicSolverFixed


def create_test_channel(length=10000.0, num_sections=21):
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


def test_small_perturbation():
    """测试1: 小扰动（20%流量变化）"""
    print("\n" + "="*70)
    print("测试1: 小扰动场景")
    print("="*70)
    
    channel = create_test_channel()
    
    # 流量从10变到12 (20%变化)
    def q_func(t):
        if t < 1800:  # 前30分钟
            return 10.0
        else:
            return 12.0
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=q_func,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    solver = PreissmannHydrodynamicSolverFixed(
        channel=channel,
        boundary_conditions=bc,
        enable_adaptive_dt=True,
        cfl_safety_factor=0.8,
        min_dt=1.0,
        max_dt=30.0,
        max_velocity=5.0,
        transition_duration=300.0,  # 5分钟过渡
    )
    
    # 使用恒定流初值
    from channel_stability.hydrodynamics.steady_flow_solver import SteadyFlowSolver
    steady_solver = SteadyFlowSolver(channel)
    steady_results = steady_solver.solve_gradually_varied_flow(
        discharge=10.0,
        downstream_depth=2.5,
    )
    
    initial_depth = np.mean(steady_results['depths'])
    initial_discharge = 10.0
    
    print(f"\n初始条件:")
    print(f"  流量: {initial_discharge:.2f} m³/s")
    print(f"  平均水深: {initial_depth:.3f} m")
    
    # 求解
    results = solver.solve(
        total_time=3600.0,  # 1小时
        dt_initial=10.0,
        initial_depth=initial_depth,
        initial_discharge=initial_discharge,
        save_interval=10,
        verbose=True,
    )
    
    return results, channel, "小扰动"


def test_medium_perturbation():
    """测试2: 中等变化（100%流量变化）"""
    print("\n" + "="*70)
    print("测试2: 中等变化场景")
    print("="*70)
    
    channel = create_test_channel()
    
    # 流量从10变到20 (100%变化)
    def q_func(t):
        if t < 1800:
            return 10.0
        else:
            return 20.0
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=q_func,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    solver = PreissmannHydrodynamicSolverFixed(
        channel=channel,
        boundary_conditions=bc,
        enable_adaptive_dt=True,
        cfl_safety_factor=0.8,
        min_dt=1.0,
        max_dt=30.0,
        max_velocity=5.0,
        transition_duration=300.0,
    )
    
    # 初值
    from channel_stability.hydrodynamics.steady_flow_solver import SteadyFlowSolver
    steady_solver = SteadyFlowSolver(channel)
    steady_results = steady_solver.solve_gradually_varied_flow(
        discharge=10.0,
        downstream_depth=2.5,
    )
    
    initial_depth = np.mean(steady_results['depths'])
    
    # 求解
    results = solver.solve(
        total_time=3600.0,
        dt_initial=10.0,
        initial_depth=initial_depth,
        initial_discharge=10.0,
        save_interval=10,
        verbose=True,
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
    
    # Courant数
    dt_mean = np.mean(results.dt_history) if results.dt_history else 0
    dx = channel.stations[1] - channel.stations[0]
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
    
    if results.dt_history:
        print(f"\n时间步长:")
        print(f"  最小值: {min(results.dt_history):.2f} s")
        print(f"  最大值: {max(results.dt_history):.2f} s")
        print(f"  平均值: {np.mean(results.dt_history):.2f} s")
    
    # 检查是否通过
    checks = {
        '流速范围': v_max < 5.0 and v_min >= 0,
        'Froude数范围': fr_max < 1.5 and fr_min >= 0,
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


def plot_comparison(results, channel, scenario_name, output_file):
    """绘制对比图"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'修复后求解器测试结果 - {scenario_name}', 
                 fontsize=16, fontweight='bold')
    
    stations = results.stations
    times_h = results.times / 3600.0
    mid_idx = len(stations) // 2
    
    # 1. 水深时间历程
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times_h, results.depths[:, mid_idx], 'b-', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title(f'Depth at station {stations[mid_idx]:.0f}m')
    ax1.grid(True, alpha=0.3)
    
    # 2. 流速时间历程
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times_h, results.velocities[:, mid_idx], 'g-', linewidth=2)
    ax2.axhline(y=5.0, color='r', linestyle='--', label='Limit')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title(f'Velocity at station {stations[mid_idx]:.0f}m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Froude数时间历程
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(times_h, results.froude_numbers[:, mid_idx], 'm-', linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Critical')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Froude Number')
    ax3.set_title(f'Froude Number at station {stations[mid_idx]:.0f}m')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 流量沿程
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(stations, results.discharges[-1, :], 'b-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Station (m)')
    ax4.set_ylabel('Discharge (m³/s)')
    ax4.set_title('Discharge Distribution (Final)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 流速沿程
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(stations, results.velocities[-1, :], 'g-o', linewidth=2, markersize=4)
    ax5.axhline(y=5.0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Station (m)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Velocity Distribution (Final)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Courant数分布
    ax6 = fig.add_subplot(gs[1, 2])
    if results.dt_history:
        dt_mean = np.mean(results.dt_history)
        dx = stations[1] - stations[0]
        courant = np.abs(results.velocities) * dt_mean / dx
        im = ax6.imshow(courant.T, aspect='auto', origin='lower',
                       extent=[times_h[0], times_h[-1], stations[0], stations[-1]],
                       cmap='RdYlGn_r', vmin=0, vmax=1.5)
        plt.colorbar(im, ax=ax6, label='Courant Number')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('Station (m)')
        ax6.set_title('Courant Number Distribution')
    
    # 7. 时间步长历史
    ax7 = fig.add_subplot(gs[2, 0])
    if results.dt_history:
        ax7.plot(range(len(results.dt_history)), results.dt_history, 'r-', linewidth=2)
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Time Step (s)')
        ax7.set_title('Adaptive Time Step History')
        ax7.grid(True, alpha=0.3)
    
    # 8. 质量守恒
    ax8 = fig.add_subplot(gs[2, 1])
    q_in = results.discharges[:, 0]
    q_out = results.discharges[:, -1]
    ax8.plot(times_h, q_in, 'b-', linewidth=2, label='Inlet')
    ax8.plot(times_h, q_out, 'r--', linewidth=2, label='Outlet')
    ax8.set_xlabel('Time (hours)')
    ax8.set_ylabel('Discharge (m³/s)')
    ax8.set_title('Mass Conservation')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 统计信息
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    v_max = np.max(results.velocities)
    v_min = np.min(results.velocities)
    fr_max = np.max(results.froude_numbers)
    mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
    
    if results.dt_history:
        dt_mean = np.mean(results.dt_history)
        dx = stations[1] - stations[0]
        courant = np.abs(results.velocities) * dt_mean / dx
        co_max = np.max(courant)
    else:
        co_max = 0
    
    text_lines = [
        "Key Statistics:",
        "",
        f"Velocity: [{v_min:.3f}, {v_max:.3f}] m/s",
        f"Froude: [0, {fr_max:.3f}]",
        f"Courant Max: {co_max:.3f}",
        f"Mass Error: {mass_error*100:.2f}%",
        "",
        "Checks:",
        f"{'✓' if v_max < 5.0 else '✗'} Velocity < 5 m/s",
        f"{'✓' if fr_max < 1.5 else '✗'} Froude < 1.5",
        f"{'✓' if co_max < 2.0 else '✗'} Courant < 2.0",
        f"{'✓' if mass_error < 0.05 else '✗'} Mass Error < 5%",
    ]
    
    text = "\n".join(text_lines)
    ax9.text(0.1, 0.5, text, fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if v_max < 5.0 else 'lightcoral', alpha=0.8))
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 图表已保存: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("修复后求解器验证测试")
    print("="*70)
    
    output_dir = './tests/fixed_solver_results'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # 测试1: 小扰动
    try:
        results1, channel1, name1 = test_small_perturbation()
        checks1, passed1 = evaluate_results(results1, channel1, name1)
        plot_comparison(results1, channel1, name1, 
                       os.path.join(output_dir, 'test1_small_perturbation.png'))
        all_results.append((name1, checks1, passed1))
    except Exception as e:
        print(f"\n✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 中等变化
    try:
        results2, channel2, name2 = test_medium_perturbation()
        checks2, passed2 = evaluate_results(results2, channel2, name2)
        plot_comparison(results2, channel2, name2,
                       os.path.join(output_dir, 'test2_medium_perturbation.png'))
        all_results.append((name2, checks2, passed2))
    except Exception as e:
        print(f"\n✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成总结报告
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
        print("✓✓✓ 所有测试通过！修复成功！")
    else:
        print("⚠ 部分测试未通过，需要进一步调整")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

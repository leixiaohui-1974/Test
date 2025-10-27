"""
测试准稳态求解器（完全守恒方案）
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
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.hydrodynamics.quasi_steady_solver import QuasiSteadySolver


def create_test_channel(length=5000.0, num_sections=11):
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


def test_scenario(scenario_name, q_before, q_after):
    """测试场景"""
    print(f"\n{'='*70}")
    print(f"测试: {scenario_name} ({q_before}→{q_after} m³/s)")
    print(f"{'='*70}\n")
    
    channel = create_test_channel()
    
    # 边界条件
    def q_func(t):
        if t < 1800:
            return q_before
        else:
            return q_after
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=q_func,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    # 创建准稳态求解器
    solver = QuasiSteadySolver(
        channel=channel,
        boundary_conditions=bc,
    )
    
    # 求解
    results = solver.solve(
        total_time=3600.0,  # 1小时
        dt=60.0,  # 1分钟时间步长
        initial_depth=2.0,
        initial_discharge=q_before,
        save_interval=1,  # 保存所有步
    )
    
    # 评价
    print(f"{'='*70}")
    print("结果评价")
    print(f"{'='*70}\n")
    
    v_max = np.max(results.velocities)
    v_min = np.min(results.velocities)
    fr_max = np.max(results.froude_numbers)
    
    q_in = results.discharges[:, 0]
    q_out = results.discharges[:, -1]
    mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
    
    dx = channel.stations[1] - channel.stations[0]
    dt_mean = 60.0
    courant = np.abs(results.velocities) * dt_mean / dx
    co_max = np.max(courant)
    
    print(f"流速: [{v_min:.3f}, {v_max:.3f}] m/s")
    print(f"Froude: [0, {fr_max:.3f}]")
    print(f"Courant: {co_max:.3f}")
    print(f"质量守恒误差: {mass_error*100:.4f}%")
    
    # 检查
    checks = {
        '流速正值': v_min >= 0,
        '流速<5m/s': v_max < 5.0,
        'Froude<1.5': fr_max < 1.5,
        'Courant<2': co_max < 2.0,
        '质量守恒<5%': mass_error < 0.05,
        '质量守恒<1%': mass_error < 0.01,
    }
    
    print(f"\n质量检查:")
    for name, passed in checks.items():
        print(f"  {'✓' if passed else '✗'} {name}")
    
    all_critical = all(checks[k] for k in ['流速正值', '流速<5m/s', 'Froude<1.5', 'Courant<2', '质量守恒<5%'])
    print(f"\n总体: {'✓ 通过' if all_critical else '✗ 未通过'}")
    
    return all_critical, results, mass_error


def plot_results(results_dict, output_file):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['b', 'g', 'r', 'm']
    
    for i, (name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        times_h = results.times / 3600.0
        
        # 水深
        mid_idx = len(results.stations) // 2
        axes[0, 0].plot(times_h, results.depths[:, mid_idx], 
                       color=color, linewidth=2, label=name)
        
        # 流速
        axes[0, 1].plot(times_h, results.velocities[:, mid_idx], 
                       color=color, linewidth=2, label=name)
        
        # 流量
        q_in = results.discharges[:, 0]
        q_out = results.discharges[:, -1]
        axes[1, 0].plot(times_h, q_in, '-', color=color, linewidth=2, label=f'{name}-inlet')
        axes[1, 0].plot(times_h, q_out, '--', color=color, linewidth=1.5, label=f'{name}-outlet')
        
        # 质量守恒误差
        mass_errors = np.abs(q_out - q_in) / (q_in + 1e-10)
        axes[1, 1].plot(times_h, mass_errors * 100, color=color, linewidth=2, label=name)
    
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Depth (m)')
    axes[0, 0].set_title('Depth at Middle Station')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity at Middle Station')
    axes[0, 1].axhline(y=5.0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Discharge (m³/s)')
    axes[1, 0].set_title('Mass Conservation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Mass Error (%)')
    axes[1, 1].set_title('Mass Conservation Error')
    axes[1, 1].axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='5% target')
    axes[1, 1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1% excellent')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 对比图已保存: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("准稳态求解器测试（完全守恒方案）")
    print("="*70)
    
    test_cases = [
        ("小幅变化", 10.0, 12.0),
        ("中等变化", 10.0, 20.0),
        ("大幅变化", 10.0, 30.0),
        ("极大变化", 10.0, 50.0),
    ]
    
    results_dict = {}
    summary = {}
    
    for name, q_before, q_after in test_cases:
        try:
            passed, results, mass_error = test_scenario(name, q_before, q_after)
            results_dict[name] = results
            summary[name] = {'passed': passed, 'mass_error': mass_error}
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            summary[name] = {'passed': False, 'mass_error': 1.0}
    
    # 绘图
    if results_dict:
        output_dir = './tests/quasi_steady_results'
        os.makedirs(output_dir, exist_ok=True)
        plot_results(results_dict, os.path.join(output_dir, 'comparison.png'))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70 + "\n")
    
    print("| 场景 | 状态 | 质量守恒误差 | 评级 |")
    print("|------|------|-------------|------|")
    for name, data in summary.items():
        status = "✓" if data['passed'] else "✗"
        error = f"{data['mass_error']*100:.4f}%"
        if data['mass_error'] < 0.01:
            grade = "A+"
        elif data['mass_error'] < 0.05:
            grade = "A"
        elif data['mass_error'] < 0.1:
            grade = "B"
        else:
            grade = "C"
        print(f"| {name} | {status} | {error} | {grade} |")
    
    passed_count = sum(1 for d in summary.values() if d['passed'])
    print(f"\n通过率: {passed_count}/{len(summary)} ({passed_count/len(summary)*100:.1f}%)")
    
    if passed_count == len(summary):
        print("\n" + "="*70)
        print("✓✓✓ 所有测试通过！准稳态求解器成功！")
        print("="*70 + "\n")
        return True
    else:
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

"""
测试智能边界条件

验证自动平滑化阶跃变化的效果
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
from channel_stability.hydrodynamics.smart_boundary_conditions import create_smart_bc
from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
from channel_stability.hydrodynamics.steady_flow_solver import SteadyFlowSolver


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


def test_discharge_step(step_magnitude: str):
    """测试流量阶跃场景"""
    
    if step_magnitude == "小幅":
        baseline_q, step_q = 10.0, 12.0
    elif step_magnitude == "中等":
        baseline_q, step_q = 10.0, 20.0
    elif step_magnitude == "大幅":
        baseline_q, step_q = 10.0, 30.0
    else:
        baseline_q, step_q = 10.0, 15.0
    
    print(f"\n{'='*70}")
    print(f"测试: 流量{step_magnitude}阶跃 ({baseline_q}→{step_q} m³/s)")
    print(f"{'='*70}\n")
    
    channel = create_test_channel()
    
    # 创建智能边界条件
    def q_func(t):
        if t < 1800:
            return baseline_q
        else:
            return step_q
    
    smart_bc = create_smart_bc(
        upstream_q_func=q_func,
        downstream_h_func=lambda t, q: 102.5,
        auto_smooth=True,  # 启用自动平滑
    )
    
    # 稳态初值
    print("计算稳态初值...")
    steady_solver = SteadyFlowSolver(channel)
    steady_results = steady_solver.solve_gradually_varied_flow(
        discharge=baseline_q,
        downstream_depth=2.5,
    )
    initial_depth = np.mean(steady_results['depths'])
    print(f"  初始水深: {initial_depth:.3f} m")
    
    # 创建求解器
    solver = PreissmannHydrodynamicSolver(
        channel=channel,
        boundary_conditions=smart_bc,  # 使用智能边界条件
        theta=0.7,
        max_iterations=30,
        relaxation=0.05,
        enable_adaptive_mesh=False,
    )
    
    # 求解
    print("\n开始模拟（智能边界条件）...")
    results = solver.solve(
        total_time=3600.0,  # 1小时
        dt=5.0,
        initial_depth=initial_depth,
        initial_discharge=baseline_q,
        save_interval=20,
        use_adaptive_dt=True,
        cfl_max=0.8,
        min_dt=1.0,
        max_dt=20.0,
    )
    
    # 评价
    print(f"\n{'='*70}")
    print("结果评价")
    print(f"{'='*70}\n")
    
    v_max = np.max(results.velocities)
    v_min = np.min(results.velocities)
    fr_max = np.max(results.froude_numbers)
    
    q_in = results.discharges[:, 0]
    q_out = results.discharges[:, -1]
    mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10))
    
    dx = channel.stations[1] - channel.stations[0]
    dt_mean = (results.times[1] - results.times[0]) if len(results.times) > 1 else 5.0
    courant = np.abs(results.velocities) * dt_mean / dx
    co_max = np.max(courant)
    
    print(f"流速: [{v_min:.3f}, {v_max:.3f}] m/s")
    print(f"Froude: [0, {fr_max:.3f}]")
    print(f"Courant: {co_max:.3f}")
    print(f"质量守恒误差: {mass_error*100:.2f}%")
    
    # 检查
    checks = {
        '流速正值': v_min >= 0,
        '流速<5m/s': v_max < 5.0,
        'Froude<1.5': fr_max < 1.5,
        'Courant<2': co_max < 2.0,
        '质量守恒<5%': mass_error < 0.05,
    }
    
    print(f"\n质量检查:")
    for name, passed in checks.items():
        print(f"  {'✓' if passed else '✗'} {name}")
    
    all_passed = all(checks.values())
    print(f"\n总体: {'✓ 通过' if all_passed else '✗ 未通过'}")
    
    return all_passed, mass_error


def main():
    """主函数"""
    print("\n" + "="*70)
    print("智能边界条件测试")
    print("="*70)
    
    test_cases = ["小幅", "中等", "大幅"]
    results = {}
    
    for magnitude in test_cases:
        try:
            passed, mass_error = test_discharge_step(magnitude)
            results[magnitude] = {'passed': passed, 'mass_error': mass_error}
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[magnitude] = {'passed': False, 'mass_error': 1.0}
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70 + "\n")
    
    print("| 变化幅度 | 状态 | 质量守恒误差 |")
    print("|---------|------|-------------|")
    for mag, data in results.items():
        status = "✓" if data['passed'] else "✗"
        error = f"{data['mass_error']*100:.2f}%" if data['mass_error'] < 0.5 else ">50%"
        print(f"| {mag} | {status} | {error} |")
    
    passed_count = sum(1 for d in results.values() if d['passed'])
    print(f"\n通过率: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")
    
    if passed_count == len(results):
        print("\n✓✓✓ 所有测试通过！智能边界条件修复成功！")
        return True
    else:
        print(f"\n⚠ {len(results) - passed_count}个测试未通过")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
